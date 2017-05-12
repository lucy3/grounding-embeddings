#include <iostream>
#include <fstream>

#include <set>
#include <string>
#include <vector>


typedef struct cooccur_rec {
    int word1;
    int word2;
    double val;
} CREC;



int load_keep(const std::string& vocab_file, const std::string& keep_file,
              std::set<int>& keep_vocab) {
    std::ifstream vocab_f(vocab_file);
    std::vector<std::string> vocab;

    std::string word;
    long freq;
    while (!vocab_f.eof()) {
        vocab_f >> word;
        vocab_f >> freq;
        vocab.push_back(word);
    }
    std::cout << "Original vocab has " << vocab.size() << " words.\n";

    // --------

    std::ifstream keep_f(keep_file);
    std::set<std::string> keep;
    while (!keep_f.eof()) {
        keep_f >> word;
        keep.insert(word);
    }

    // Retain only keep words
    for (int i = 0; i < vocab.size(); ++i) {
        if (keep.find(vocab[i]) != keep.end()) {
            keep_vocab.insert(i);
        }
    }

    std::cout << "Filtered vocab has " << keep_vocab.size() << " words.\n";
}


int filter_binary(const std::string& input_file, const std::string& output_file,
                  std::set<int>& keep_vocab) {
    std::ifstream file(input_file, std::ios::binary);
    std::ofstream o_file(output_file, std::ios::binary);
    CREC crec;

    long long n = 0, kept = 0;
    while (file.read(reinterpret_cast<char*>(&crec), sizeof crec)) {
        if (file.eof()) break;
        n++;

        bool keep = (keep_vocab.find(crec.word1 - 1) != keep_vocab.end())
            && (keep_vocab.find(crec.word2 - 1) != keep_vocab.end());
        if (keep) {
            kept += 1;
            o_file.write(reinterpret_cast<char*>(&crec), sizeof crec);
        }

        if (n % 100000000 == 0)
            std::cout << n << "\n";
    }

    std::cout << "Kept " << kept << " of " << n << " elements.\n";
}


int main(int argc, char *argv[]) {
    std::set<int> keep;

    std::string datadir = argv[1];

    auto vocab_file = datadir + "/vocab.txt";
    auto keep_file = datadir + "/vocab.keep.txt";
    load_keep(vocab_file, keep_file, keep);

    auto input_file = datadir + "/cooccurrence.bin";
    auto output_file = datadir + "/cooccurrence.filtered.bin";
    filter_binary(input_file, output_file, keep);
}
