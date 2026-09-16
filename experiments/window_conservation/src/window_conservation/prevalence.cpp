#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

// Fixed-width records. Hashing is bijective on uint64_t; canonical k <= 31.
struct Counts {
    uint32_t last = 0;
    uint16_t species = 0, current = 0, maximum = 0;
};
struct Entry {
    uint64_t key;
    uint32_t species;
    uint16_t maximum, reserved;
};
static_assert(sizeof(Entry) == 16);
uint64_t mix(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}
int basecode(char c) {
    switch (c) {
        case 'A': case 'a': return 0;
        case 'C': case 'c': return 1;
        case 'G': case 'g': return 2;
        case 'T': case 't': return 3;
        default: return -1;
    }
}
struct Rolling {
    int k, valid = 0;
    uint64_t f = 0, r = 0, mask;
    explicit Rolling(int length) : k(length), mask((1ULL << (2*length))-1) {}
    void clear() { valid = 0; f = r = 0; }
    bool push(char c, uint64_t &key) {
        int b = basecode(c);
        if (b < 0) { clear(); return false; }
        f = ((f << 2) | b) & mask;
        r = (r >> 2) | (uint64_t(3-b) << (2*(k-1)));
        valid = std::min(valid+1, k);
        key = mix(std::min(f,r) ^ 577ULL);
        return valid >= k;
    }
};
template<class Header, class Base> void fasta(const std::string &path, Header header, Base base, uint64_t limit=0) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("cannot read " + path);
    std::string line;
    bool have_header = false;
    uint64_t read_bases = 0;
    while (std::getline(input,line)) {
        if (!line.empty() && line[0] == '>') {
            header(line.substr(1,line.find_first_of(" \t\r")-1));
            have_header = true;
        } else {
            for (char c : line) {
                if (c == '\r' || c == ' ' || c == '\t') continue;
                if (!have_header) throw std::runtime_error("FASTA sequence before header");
                base(c);
                if (++read_bases == limit) return;
            }
        }
    }
}
void build(int k, int bits, const std::string &list, const std::string &output, uint64_t limit,
           const std::string &query = "") {
    std::ifstream paths(list);
    if (!paths) throw std::runtime_error("missing species list");
    std::unordered_map<uint64_t,Counts> counts;
    counts.reserve(1000000);
    // A query restriction changes only the keys retained, never their species/copy counts.
    if (!query.empty()) {
        Rolling rolling(k);
        fasta(query, [&](const std::string &) { rolling.clear(); }, [&](char c) {
            uint64_t key;
            if (rolling.push(c,key) && !(key & ((1ULL<<bits)-1))) counts.try_emplace(key);
        });
    }
    uint64_t bases = 0, selected = 0, valid = 0;
    uint32_t species = 0;
    std::string path;
    auto start = std::chrono::steady_clock::now();
    while (std::getline(paths,path)) {
        if (path.empty()) continue;
        if (++species > 65535) throw std::runtime_error("too many species");
        Rolling rolling(k);
        fasta(path, [&](const std::string &) { rolling.clear(); }, [&](char c) {
            ++bases;
            uint64_t key;
            if (!rolling.push(c,key)) return;
            ++valid;
            if (key & ((1ULL<<bits)-1)) return;
            ++selected;
            if (!query.empty() && counts.find(key) == counts.end()) return;
            auto &v = counts[key];
            if (v.last != species) {
                v.last = species;
                ++v.species;
                v.current = 1;
            } else if (v.current < 65535) ++v.current;
            v.maximum = std::max(v.maximum,v.current);
        },limit);
        std::cerr << "species " << species << " bases " << bases << " unique " << counts.size() << std::endl;
    }
    if (!species) throw std::runtime_error("empty species list");
    std::ofstream index(output,std::ios::binary);
    if (!index) throw std::runtime_error("cannot write index");
    for (const auto &[key,value] : counts) {
        Entry entry{key,value.species,value.maximum,0};
        index.write(reinterpret_cast<const char *>(&entry),sizeof(Entry));
    }
    if (!index) throw std::runtime_error("index write failed");
    double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
    std::cout << "{\"species\":" << species << ",\"bases\":" << bases
              << ",\"valid_kmers\":" << valid << ",\"selected_occurrences\":" << selected
              << ",\"unique_keys\":" << counts.size() << ",\"index_bytes\":" << counts.size()*sizeof(Entry)
              << ",\"seconds\":" << seconds << "}" << std::endl;
}
std::unordered_map<uint64_t,Entry> load_index(const std::string &indexpath) {
    std::ifstream index(indexpath,std::ios::binary | std::ios::ate);
    if (!index || index.tellg() % sizeof(Entry)) throw std::runtime_error("invalid index");
    std::unordered_map<uint64_t,Entry> entries;
    entries.reserve(size_t(index.tellg())/sizeof(Entry));
    index.seekg(0);
    Entry entry;
    while (index.read(reinterpret_cast<char *>(&entry),sizeof(Entry))) entries.emplace(entry.key,entry);
    return entries;
}
void score(int k, int bits, int species, int width, const std::unordered_map<uint64_t,Entry> &entries,
           const std::string &path, const std::string &output, uint64_t limit) {
    std::ofstream out(output);
    if (!out) throw std::runtime_error("cannot write scores");
    out << "chrom\tstart\tend\tvalid\tgc\trepeat\tentropy\tseeds\tany\tboth\tbreadth\tany_copy4\tboth_copy4\tbreadth_copy4\n";
    Rolling rolling(k);
    std::string chrom;
    uint64_t pos=0, window_count=0, lookups=0;
    std::vector<uint64_t> keys;
    std::array<int,4> freq{};
    int repeats=0, valid=0;
    auto start = std::chrono::steady_clock::now();
    fasta(path,[&](const std::string &name) {
        chrom=name; pos=0; rolling.clear(); keys.clear(); freq={}; repeats=valid=0;
    },[&](char c) {
        int code=basecode(c);
        if (code >= 0) { ++freq[code]; ++valid; }
        repeats += c >= 'a' && c <= 'z';
        uint64_t key;
        // Assign selected k-mers by their end position. This includes boundary-crossing evidence.
        if (rolling.push(c,key) && !(key & ((1ULL<<bits)-1))) keys.push_back(key);
        ++pos;
        if (pos % width) return;
        std::sort(keys.begin(),keys.end());
        keys.erase(std::unique(keys.begin(),keys.end()),keys.end());
        double any=0,both=0,breadth=0,any4=0,both4=0,breadth4=0;
        for (uint64_t key : keys) {
            ++lookups;
            auto it=entries.find(key);
            if (it==entries.end()) throw std::runtime_error("scored sequence absent from index");
            double others=it->second.species-1;
            any += others>=1; both += others>=2; breadth += others/std::max(1,species-1);
            if (it->second.maximum<=4) {
                any4 += others>=1; both4 += others>=2; breadth4 += others/std::max(1,species-1);
            }
        }
        double denom=std::max(size_t(1),keys.size()),entropy=0;
        for (int n:freq) if (n) { double p=double(n)/std::max(1,valid); entropy-=p*std::log2(p); }
        out << chrom << '\t' << pos-width << '\t' << pos << '\t' << valid << '\t'
            << double(freq[1]+freq[2])/std::max(1,valid) << '\t' << double(repeats)/width << '\t'
            << entropy << '\t' << keys.size() << '\t' << any/denom << '\t' << both/denom << '\t'
            << breadth/denom << '\t' << any4/denom << '\t' << both4/denom << '\t' << breadth4/denom << '\n';
        ++window_count;
        keys.clear(); freq={}; repeats=valid=0;
    },limit);
    if (!out) throw std::runtime_error("score write failed");
    double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
    std::cout << "{\"windows\":" << window_count << ",\"lookups\":" << lookups
              << ",\"seconds\":" << seconds << "}" << std::endl;
}
int main(int argc,char **argv) {
    try {
        if (argc<6) throw std::runtime_error("build K BITS LIST INDEX | score K BITS SPECIES WIDTH INDEX FASTA OUT");
        std::string mode=argv[1]; int k=std::stoi(argv[2]),bits=std::stoi(argv[3]);
        if (k<1 || k>31 || bits<0 || bits>20) throw std::runtime_error("invalid k or sampling");
        if (mode=="build" && (argc==6 || argc==7)) build(k,bits,argv[4],argv[5],argc==7 ? std::stoull(argv[6]) : 0);
        else if (mode=="build-query" && argc==7) build(k,bits,argv[4],argv[5],0,argv[6]);
        else if ((mode=="score" || mode=="score-list") && (argc==9 || argc==10)) {
            int species=std::stoi(argv[4]),width=std::stoi(argv[5]);
            if (species<1 || species>65535 || width<k) throw std::runtime_error("invalid score geometry");
            uint64_t limit=argc==10 ? std::stoull(argv[9]) : 0;
            auto entries=load_index(argv[6]);
            if (mode=="score") score(k,bits,species,width,entries,argv[7],argv[8],limit);
            else {
                std::ifstream input(argv[7]);
                if (!input) throw std::runtime_error("missing score list");
                std::filesystem::create_directories(argv[8]);
                std::string path; int seen=0;
                while (std::getline(input,path)) if (!path.empty()) {
                    ++seen;
                    auto output=std::filesystem::path(argv[8])/(std::filesystem::path(path).stem().string()+".tsv");
                    score(k,bits,species,width,entries,path,output.string(),limit);
                }
                if (seen!=species) throw std::runtime_error("species count mismatch");
            }
        } else throw std::runtime_error("invalid arguments");
        return 0;
    } catch (const std::exception &error) { std::cerr << error.what() << std::endl; return 1; }
}
