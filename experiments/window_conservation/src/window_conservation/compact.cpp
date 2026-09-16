// Exact compact counting experiment; share the validated rolling DNA parser.
#define main baseline_main
#include "prevalence.cpp"
#undef main
#include <sys/resource.h>
#include <sstream>

struct CompactValue {
    uint16_t last=0, species=0, current=0, maximum=0;
};
struct Slot {
    uint64_t key=0;
    CompactValue value{65535,0,0,0};
    bool empty() const { return value.last==65535 && value.species==0; }
};
static_assert(sizeof(Slot)==16);
class Flat {
    std::vector<Slot> slots;
    size_t used=0;
    size_t locate(uint64_t key) const {
        // Sampling fixes low bits of key, so hash again for the bucket.
        size_t pos=mix(key)&(slots.size()-1);
        while (!slots[pos].empty() && slots[pos].key!=key) pos=(pos+1)&(slots.size()-1);
        return pos;
    }
    void grow() {
        auto old=std::move(slots);
        slots.resize(std::max(size_t(1024),old.size()*2));
        for (const auto &slot:old) if (!slot.empty()) slots[locate(slot.key)]=slot;
    }
public:
    Flat() { grow(); }
    size_t size() const { return used; }
    size_t bytes() const { return slots.size()*sizeof(Slot); }
    CompactValue *find(uint64_t key) {
        auto &slot=slots[locate(key)];
        return slot.empty() ? nullptr : &slot.value;
    }
    const CompactValue *find(uint64_t key) const {
        const auto &slot=slots[locate(key)];
        return slot.empty() ? nullptr : &slot.value;
    }
    CompactValue &insert(uint64_t key) {
        if ((used+1)*5>slots.size()*4) grow();
        auto &slot=slots[locate(key)];
        if (slot.empty()) { slot.key=key; slot.value={}; ++used; }
        return slot.value;
    }
    template<class F> void each(F callback) const {
        for (const auto &slot:slots) if (!slot.empty()) callback(slot.key,slot.value);
    }
};
long peak_rss() { struct rusage r{}; getrusage(RUSAGE_SELF,&r); return r.ru_maxrss; }
void compact_score(int k,int bits,int species,int width,const Flat &entries,
                   const std::string &path,const std::string &output,uint64_t limit=0,int bottom=0) {
    std::ofstream out(output);
    if (!out) throw std::runtime_error("cannot write scores");
    out << "chrom\tstart\tend\tvalid\tgc\trepeat\tentropy\tseeds\tany\tboth\tbreadth\tany_copy4\tboth_copy4\tbreadth_copy4\n";
    Rolling rolling(k);
    std::string chrom;
    uint64_t pos=0,windows=0,lookups=0;
    std::vector<uint64_t> keys;
    std::array<int,4> freq{};
    int repeats=0,valid=0;
    auto start=std::chrono::steady_clock::now();
    fasta(path,[&](const std::string &name) {
        chrom=name; pos=0; rolling.clear(); keys.clear(); freq={}; repeats=valid=0;
    },[&](char c) {
        int code=basecode(c);
        if (code>=0) { ++freq[code]; ++valid; }
        repeats+=c>='a' && c<='z';
        uint64_t key;
        if (rolling.push(c,key) && (bottom || !(key&((1ULL<<bits)-1)))) keys.push_back(key);
        if (++pos%width) return;
        std::sort(keys.begin(),keys.end());
        keys.erase(std::unique(keys.begin(),keys.end()),keys.end());
        if (bottom && keys.size()>size_t(bottom)) keys.resize(bottom);
        double any=0,both=0,breadth=0,any4=0,both4=0,breadth4=0;
        for (uint64_t word:keys) {
            ++lookups;
            const auto *v=entries.find(word);
            if (!v || !v->species) throw std::runtime_error("query word absent");
            double others=v->species-1;
            any+=others>=1; both+=others>=2; breadth+=others/std::max(1,species-1);
            if (v->maximum<=4) { any4+=others>=1; both4+=others>=2; breadth4+=others/std::max(1,species-1); }
        }
        double denom=std::max(size_t(1),keys.size()),entropy=0;
        for (int n:freq) if (n) { double p=double(n)/std::max(1,valid); entropy-=p*std::log2(p); }
        out << chrom << '\t' << pos-width << '\t' << pos << '\t' << valid << '\t'
            << double(freq[1]+freq[2])/std::max(1,valid) << '\t' << double(repeats)/width << '\t'
            << entropy << '\t' << keys.size() << '\t' << any/denom << '\t' << both/denom << '\t'
            << breadth/denom << '\t' << any4/denom << '\t' << both4/denom << '\t' << breadth4/denom << '\n';
        ++windows; keys.clear(); freq={}; repeats=valid=0;
    },limit);
    if (!out) throw std::runtime_error("score write failed");
    std::cout << "{\"stage\":\"score\",\"species\":" << species << ",\"width\":" << width
              << ",\"bits\":" << bits << ",\"bottom\":" << bottom << ",\"windows\":" << windows << ",\"lookups\":" << lookups
              << ",\"seconds\":" << std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()
              << ",\"peak_rss_kib\":" << peak_rss() << "}" << std::endl;
}
int main(int argc,char **argv) {
    try {
        if (argc<8 || argc>10) throw std::runtime_error("compact K BITS LIST QUERY OUTDIR PANELS WIDTH [LIMIT] [BOTTOM]");
        int k=std::stoi(argv[1]),bits=std::stoi(argv[2]),width=std::stoi(argv[7]);
        uint64_t limit=argc>=9 ? std::stoull(argv[8]) : 0;
        int bottom=argc==10 ? std::stoi(argv[9]) : 0;
        if (bottom<0 || bottom>width) throw std::runtime_error("invalid bottom size");
        if (k<1 || k>31 || bits<0 || bits>20 || width<k) throw std::runtime_error("invalid geometry");
        std::string query=argv[4];
        if (bottom && query=="-") throw std::runtime_error("bottom sketch requires explicit query universe");
        std::vector<int> panels;
        std::string part; std::istringstream panel_stream(argv[6]);
        while (std::getline(panel_stream,part,',')) panels.push_back(std::stoi(part));
        if (panels.empty() || *std::min_element(panels.begin(),panels.end())<1) throw std::runtime_error("invalid panels");
        std::filesystem::create_directories(argv[5]);
        Flat counts;
        auto start=std::chrono::steady_clock::now();
        if (query!="-") {
            Rolling rolling(k);
            std::vector<uint64_t> keys; uint64_t pos=0;
            fasta(query,[&](const std::string &) { rolling.clear(); keys.clear(); pos=0; },[&](char c) {
                uint64_t key;
                if (rolling.push(c,key)) {
                    if (bottom) keys.push_back(key);
                    else if (!(key&((1ULL<<bits)-1))) counts.insert(key);
                }
                if (++pos%width==0 && bottom) {
                    std::sort(keys.begin(),keys.end());
                    keys.erase(std::unique(keys.begin(),keys.end()),keys.end());
                    if (keys.size()>size_t(bottom)) keys.resize(bottom);
                    for (uint64_t word:keys) counts.insert(word);
                    keys.clear();
                }
            });
        }
        std::vector<uint64_t> bloom;
        auto bloom_bits=[](uint64_t key) {
            return (1ULL<<((key>>2)&63)) | (1ULL<<((key>>8)&63)) | (1ULL<<((key>>14)&63));
        };
        if (query!="-") {
            size_t blocks=1;
            while (blocks<counts.size()/8) blocks*=2;
            bloom.resize(blocks);
            counts.each([&](uint64_t key,const CompactValue &) { bloom[(key>>20)&(blocks-1)]|=bloom_bits(key); });
        }
        if (std::string(argv[3])=="-") {
            if (query=="-") throw std::runtime_error("profile requires query");
            std::cout << "{\"stage\":\"query_profile\",\"bits\":" << bits << ",\"bottom\":" << bottom
                      << ",\"unique_keys\":" << counts.size() << ",\"table_bytes\":" << counts.bytes()
                      << ",\"bloom_bytes\":" << bloom.size()*sizeof(uint64_t) << ",\"peak_rss_kib\":" << peak_rss()
                      << ",\"seconds\":" << std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count() << "}" << std::endl;
            return 0;
        }
        std::ifstream listing(argv[3]);
        if (!listing) throw std::runtime_error("missing species list");
        std::string path; int species=0;
        uint64_t bases=0,selected=0;
        double scoring_seconds=0;
        std::vector<std::string> seen_paths;
        while (std::getline(listing,path)) if (!path.empty()) {
            if (++species>65535) throw std::runtime_error("too many species");
            seen_paths.push_back(path);
            Rolling rolling(k);
            fasta(path,[&](const std::string &) { rolling.clear(); },[&](char c) {
                ++bases; uint64_t key;
                if (!rolling.push(c,key) || (!bottom && (key&((1ULL<<bits)-1)))) return;
                ++selected;
                CompactValue *value;
                if (query!="-") {
                    uint64_t mask=bloom_bits(key);
                    if ((bloom[(key>>20)&(bloom.size()-1)]&mask)!=mask) return;
                    value=counts.find(key);
                    if (!value) return;
                } else value=&counts.insert(key);
                auto &v=*value;
                if (v.last!=species) { v.last=species; ++v.species; v.current=1; }
                else if (v.current<65535) ++v.current;
                v.maximum=std::max(v.maximum,v.current);
            },limit);
            std::cerr << "species " << species << " bases " << bases << " unique " << counts.size() << std::endl;
            if (std::find(panels.begin(),panels.end(),species)==panels.end()) continue;
            std::cout << "{\"stage\":\"build\",\"species\":" << species << ",\"bases\":" << bases
                      << ",\"selected_occurrences\":" << selected << ",\"unique_keys\":" << counts.size()
                      << ",\"table_bytes\":" << counts.bytes() << ",\"peak_rss_kib\":" << peak_rss()
                      << ",\"seconds\":" << std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()-scoring_seconds << "}" << std::endl;
            auto scoring_start=std::chrono::steady_clock::now();
            auto output=std::filesystem::path(argv[5])/std::to_string(species);
            std::filesystem::create_directories(output);
            if (query!="-") {
                compact_score(k,bits,species,width,counts,query,(output/"human.tsv").string(),0,bottom);
                if (bottom==32) compact_score(k,bits,species,width,counts,query,(output/"bottom16.tsv").string(),0,16);
                if (!bottom && bits==2) for (int thin: {3,4})
                    compact_score(k,thin,species,width,counts,query,(output/("bits"+std::to_string(thin)+".tsv")).string());
            }
            else for (const auto &p:seen_paths)
                compact_score(k,bits,species,width,counts,p,(output/(std::filesystem::path(p).stem().string()+".tsv")).string(),limit);
            scoring_seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-scoring_start).count();
        }
        if (species==0 || *std::max_element(panels.begin(),panels.end())>species) throw std::runtime_error("missing panel species");
        return 0;
    } catch (const std::exception &e) { std::cerr << e.what() << std::endl; return 1; }
}
