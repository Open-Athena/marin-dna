// Disk-partitioned exact global scoring; one input scan and bounded window chunks.
#define main baseline_main
#include "prevalence.cpp"
#undef main
#include <sstream>
#include <sys/resource.h>

constexpr uint64_t invalid_window=(1ULL<<48)-1;
#ifndef WINDOW_CHUNK_SIZE
#define WINDOW_CHUNK_SIZE 262144
#endif
constexpr uint64_t chunk_windows=WINDOW_CHUNK_SIZE;
static_assert(chunk_windows>0);
struct Occurrence { uint64_t key, location; };
struct Contribution { uint64_t window; std::array<uint32_t,6> values{}; };
static_assert(sizeof(Occurrence)==16 && sizeof(Contribution)==32);
template<class T> void put(std::ofstream &out,const T &value) {
    out.write(reinterpret_cast<const char *>(&value),sizeof(value));
    if (!out) throw std::runtime_error("scratch write failed");
}
int main(int argc,char **argv) {
    try {
        if (argc!=8 && argc!=9) throw std::runtime_error("partition K BITS WIDTH PARTITIONS LIST DIRECTORY OUTPUT [LIMIT]");
        int k=std::stoi(argv[1]),bits=std::stoi(argv[2]),width=std::stoi(argv[3]),parts=std::stoi(argv[4]);
        uint64_t limit=argc==9 ? std::stoull(argv[8]) : 0;
        if (k<1 || k>31 || bits<0 || bits>20 || width<k || width>4096 || parts<1 || parts>512 || (parts&(parts-1))) throw std::runtime_error("invalid geometry");
        std::filesystem::path scratch=argv[6];
        if (std::filesystem::exists(scratch)) throw std::runtime_error("scratch directory must be new");
        std::filesystem::create_directories(scratch);
        std::vector<std::ofstream> shards;
        for (int i=0;i<parts;++i) shards.emplace_back(scratch/("words-"+std::to_string(i)),std::ios::binary);
        std::ofstream metadata(scratch/"metadata.tsv");
        if (!metadata) throw std::runtime_error("cannot write metadata");
        std::ifstream listing(argv[5]);
        if (!listing) throw std::runtime_error("cannot read species list");
        uint64_t window=0,bases=0,occurrences=0;
        int species=0;
        std::string path;
        auto started=std::chrono::steady_clock::now();
        while (std::getline(listing,path)) if (!path.empty()) {
            if (++species>65535) throw std::runtime_error("too many species");
            Rolling rolling(k);
            std::string chrom;
            uint64_t pos=0;
            std::vector<uint64_t> keys;
            std::array<int,4> freq{};
            int repeats=0,valid=0;
            auto flush=[&](bool complete) {
                uint64_t id=complete ? window : invalid_window;
                if (window>=invalid_window) throw std::runtime_error("too many windows");
                for (uint64_t key:keys) {
                    put(shards[(key>>32)&(parts-1)],Occurrence{key,(id<<16)|uint64_t(species)});
                    ++occurrences;
                }
                if (complete) {
                    std::sort(keys.begin(),keys.end());
                    keys.erase(std::unique(keys.begin(),keys.end()),keys.end());
                    double entropy=0;
                    for (int n:freq) if (n) { double p=double(n)/std::max(1,valid); entropy-=p*std::log2(p); }
                    metadata << species << '\t' << chrom << '\t' << pos-width << '\t' << pos << '\t' << valid << '\t'
                             << double(freq[1]+freq[2])/std::max(1,valid) << '\t' << double(repeats)/width << '\t' << entropy << '\t' << keys.size() << '\n';
                    ++window;
                }
                keys.clear(); freq={}; repeats=valid=0;
            };
            fasta(path,[&](const std::string &name) { flush(false); chrom=name; pos=0; rolling.clear(); },[&](char c) {
                ++bases; ++pos;
                int b=basecode(c);
                if (b>=0) { ++freq[b]; ++valid; }
                repeats+=c>='a' && c<='z';
                uint64_t key;
                if (rolling.push(c,key) && !(key&((1ULL<<bits)-1))) keys.push_back(key);
                if (pos%width==0) flush(true);
            },limit);
            flush(false);
        }
        if (!species) throw std::runtime_error("no species");
        for (auto &shard:shards) { shard.close(); if (!shard) throw std::runtime_error("word spool close failed"); }
        shards.clear(); metadata.close();
        if (!metadata) throw std::runtime_error("metadata write failed");
        double partition_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();
        uint64_t max_keys=0,unique_keys=0,contribution_records=0;
        auto word_started=std::chrono::steady_clock::now();
        for (int part=0;part<parts;++part) {
            auto wordpath=scratch/("words-"+std::to_string(part));
            if (std::filesystem::file_size(wordpath)%sizeof(Occurrence)) throw std::runtime_error("truncated words");
            std::unordered_map<uint64_t,Counts> counts;
            counts.reserve(std::min(uint64_t(1000000),std::filesystem::file_size(wordpath)/sizeof(Occurrence)));
            std::ifstream input(wordpath,std::ios::binary);
            Occurrence record;
            while (input.read(reinterpret_cast<char *>(&record),sizeof(record))) {
                uint32_t s=record.location&65535;
                auto &v=counts[record.key];
                if (v.last!=s) { v.last=s; ++v.species; v.current=1; }
                else if (v.current<65535) ++v.current;
                v.maximum=std::max(v.maximum,v.current);
            }
            if (!input.eof()) throw std::runtime_error("word spool read failed");
            max_keys=std::max(max_keys,uint64_t(counts.size())); unique_keys+=counts.size();
            input.clear(); input.seekg(0);
            std::vector<uint64_t> keys;
            uint64_t current=invalid_window,open_chunk=invalid_window;
            std::ofstream contributions;
            auto flush=[&]() {
                if (current==invalid_window) return;
                std::sort(keys.begin(),keys.end());
                keys.erase(std::unique(keys.begin(),keys.end()),keys.end());
                Contribution contribution{current,{}};
                for (uint64_t key:keys) {
                    const auto &v=counts.at(key);
                    uint32_t others=v.species-1;
                    contribution.values[0]+=others>=1;
                    contribution.values[1]+=others>=2;
                    contribution.values[2]+=others;
                    if (v.maximum<=4) {
                        contribution.values[3]+=others>=1;
                        contribution.values[4]+=others>=2;
                        contribution.values[5]+=others;
                    }
                }
                if (contribution.values[0]) {
                    uint64_t chunk=current/chunk_windows;
                    if (chunk!=open_chunk) {
                        contributions.close(); contributions.clear();
                        contributions.open(scratch/("windows-"+std::to_string(chunk)),std::ios::binary|std::ios::app);
                        open_chunk=chunk;
                    }
                    put(contributions,contribution); ++contribution_records;
                }
                keys.clear();
            };
            while (input.read(reinterpret_cast<char *>(&record),sizeof(record))) {
                uint64_t id=record.location>>16;
                if (id==invalid_window) continue;
                if (id!=current) { flush(); current=id; }
                keys.push_back(record.key);
            }
            if (!input.eof()) throw std::runtime_error("word spool replay failed");
            flush(); input.close(); contributions.close();
            std::filesystem::remove(wordpath);
            std::cerr << "partition " << part << " unique " << counts.size() << std::endl;
        }
        double reduce_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-word_started).count();
        std::ifstream meta(scratch/"metadata.tsv");
        std::ofstream output(argv[7]);
        if (!output) throw std::runtime_error("cannot write scores");
        output << "species\tchrom\tstart\tend\tvalid\tgc\trepeat\tentropy\tseeds\tany\tboth\tbreadth\tany_copy4\tboth_copy4\tbreadth_copy4\n";
        for (uint64_t first=0;first<window;first+=chunk_windows) {
            uint64_t size=std::min(chunk_windows,window-first);
            std::vector<std::array<uint32_t,6>> totals(size);
            auto filename=scratch/("windows-"+std::to_string(first/chunk_windows));
            if (std::filesystem::exists(filename)) {
                if (std::filesystem::file_size(filename)%sizeof(Contribution)) throw std::runtime_error("truncated contributions");
                std::ifstream input(filename,std::ios::binary);
                Contribution contribution;
                while (input.read(reinterpret_cast<char *>(&contribution),sizeof(contribution))) {
                    if (contribution.window<first || contribution.window>=first+size) throw std::runtime_error("window partition mismatch");
                    for (int j=0;j<6;++j) totals[contribution.window-first][j]+=contribution.values[j];
                }
                if (!input.eof()) throw std::runtime_error("contribution read failed");
                input.close(); std::filesystem::remove(filename);
            }
            for (uint64_t i=0;i<size;++i) {
                std::string line;
                if (!std::getline(meta,line)) throw std::runtime_error("missing window metadata");
                int seeds=std::stoi(line.substr(line.find_last_of('\t')+1));
                output << line;
                for (int j=0;j<6;++j) {
                    double denom=std::max(1,seeds);
                    if (j==2 || j==5) denom*=std::max(1,species-1);
                    output << '\t' << totals[i][j]/denom;
                }
                output << '\n';
            }
        }
        if (!output) throw std::runtime_error("score output failed");
        struct rusage usage{}; getrusage(RUSAGE_SELF,&usage);
        std::cout << "{\"bases\":" << bases << ",\"species\":" << species << ",\"windows\":" << window
                  << ",\"unique_keys\":" << unique_keys << ",\"maximum_partition_keys\":" << max_keys
                  << ",\"partitions\":" << parts << ",\"occurrence_records\":" << occurrences
                  << ",\"contribution_records\":" << contribution_records
                  << ",\"word_spool_bytes\":" << occurrences*sizeof(Occurrence)
                  << ",\"contribution_spool_bytes\":" << contribution_records*sizeof(Contribution)
                  << ",\"metadata_bytes\":" << std::filesystem::file_size(scratch/"metadata.tsv")
                  << ",\"partition_seconds\":" << partition_seconds << ",\"reduce_seconds\":" << reduce_seconds
                  << ",\"total_seconds\":" << std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count()
                  << ",\"peak_rss_kib\":" << usage.ru_maxrss << "}" << std::endl;
        meta.close(); std::filesystem::remove(scratch/"metadata.tsv"); std::filesystem::remove(scratch);
        return 0;
    } catch (const std::exception &e) { std::cerr << e.what() << std::endl; return 1; }
}
