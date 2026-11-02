// ============================================================
//  PAQ - Per-page Adaptive Q-learning for Page Replacement
//  Pious Bansal (24BCE0421) | Urvika Lamba (24BCE0348)
//  BCSE303L Operating Systems - Review 2
//
//  HOW TO RUN:
//    Online (iPad/phone): paste into onlinegdb.com, pick C++, press Run.
//    On a PC:  g++ -std=c++17 -O2 paq.cpp -o paq
//              ./paq
//  Nothing to install. No libraries. Runs in a few seconds.
// ============================================================

#include <iostream>     // cout - printing
#include <vector>       // vector - resizable array
#include <deque>        // deque - queue you can push/pop at both ends
#include <map>          // map - key -> value lookup
#include <algorithm>    // find, max, min
#include <random>       // random number generation
#include <cmath>        // log2, floor
#include <iomanip>      // setw - column alignment when printing

using namespace std;

// ------------------------------------------------------------
// SECTION 0: basic settings
// ------------------------------------------------------------
const int FRAMES   = 8;        // how many pages fit in RAM at once
const int R_BUCKET = 8;        // number of recency groups
const int F_BUCKET = 8;        // number of frequency groups
const long long INF = 1e18;    // "never used again"

// ------------------------------------------------------------
// SECTION 1: WORKLOADS  (this is our DATASET - see report Sec 4)
// A "trace" is just a list of page numbers a program touches.
// ------------------------------------------------------------

// Pattern A: SCAN. Read page 0,1,2,3... never come back. Nobody can win.
vector<int> makeScan(int length) {
    vector<int> t;
    for (int i = 0; i < length; i++) t.push_back(i);
    return t;
}

// Pattern B: LOOP. Go round a fixed set of pages forever: 0,1,2,...,L-1,0,1,2...
// If L is bigger than FRAMES, LRU gets 0% hits. This is LRU's nightmare.
vector<int> makeLoop(int cycleLength, int length) {
    vector<int> t;
    for (int i = 0; i < length; i++) t.push_back(i % cycleLength);
    return t;
}

// Pattern C: ZIPF. A few pages are very popular, most are rare.
// This is what normal programs look like. LRU and LFU do well here.
vector<int> makeZipf(int numPages, int length, double alpha, mt19937 &rng) {
    vector<double> cdf(numPages);
    double sum = 0;
    for (int i = 0; i < numPages; i++) { sum += 1.0 / pow(i + 1, alpha); cdf[i] = sum; }
    for (int i = 0; i < numPages; i++) cdf[i] /= sum;   // turn into probabilities

    uniform_real_distribution<double> pick(0.0, 1.0);
    vector<int> t;
    for (int i = 0; i < length; i++) {
        double u = pick(rng);
        int page = 0;
        while (page < numPages - 1 && cdf[page] < u) page++;
        t.push_back(page);
    }
    return t;
}

// Pattern D: PHASE-SHIFT. Zipf, then Loop, then Scan - repeated.
// The program's PERSONALITY changes mid-run. This is our main test.
// The loop phase must stay slightly LARGER than RAM to remain adversarial,
// so the number of frames is passed in rather than hardcoded.
vector<int> makePhaseShiftFor(int frames, int phaseLength, int cycles, mt19937 &rng) {
    vector<int> t;
    for (int c = 0; c < cycles; c++) {
        vector<int> a = makeZipf(40, phaseLength, 0.9, rng);
        vector<int> b = makeLoop(frames + 1, phaseLength);
        vector<int> d = makeScan(phaseLength);
        t.insert(t.end(), a.begin(), a.end());
        t.insert(t.end(), b.begin(), b.end());
        t.insert(t.end(), d.begin(), d.end());
    }
    return t;
}

vector<int> makePhaseShift(int phaseLength, int cycles, mt19937 &rng) {
    return makePhaseShiftFor(FRAMES, phaseLength, cycles, rng);
}

// ------------------------------------------------------------
// SECTION 2: THE CLASSIC POLICIES (our baselines)
// Each function returns how many page faults it caused.
// ------------------------------------------------------------

// FIFO - throw out whoever arrived first. Ignores whether it was just used.
long long runFIFO(vector<int> &refs, int frames) {
    vector<int> ram;
    long long faults = 0;
    for (int i = 0; i < (int)refs.size(); i++) {
        int page = refs[i];
        if (find(ram.begin(), ram.end(), page) != ram.end()) continue;  // HIT
        faults++;                                                        // MISS
        if ((int)ram.size() == frames) ram.erase(ram.begin());           // oldest leaves
        ram.push_back(page);
    }
    return faults;
}

// LRU - throw out whoever was used longest ago. A HIT refreshes the page.
long long runLRU(vector<int> &refs, int frames) {
    vector<int> ram;
    long long faults = 0;
    for (int i = 0; i < (int)refs.size(); i++) {
        int page = refs[i];
        vector<int>::iterator it = find(ram.begin(), ram.end(), page);
        if (it != ram.end()) {          // HIT: move page to the "fresh" end
            ram.erase(it);
            ram.push_back(page);
            continue;
        }
        faults++;
        if ((int)ram.size() == frames) ram.erase(ram.begin());
        ram.push_back(page);
    }
    return faults;
}

// MRU - the opposite bet: throw out the page used MOST recently.
long long runMRU(vector<int> &refs, int frames) {
    vector<int> ram;
    long long faults = 0;
    for (int i = 0; i < (int)refs.size(); i++) {
        int page = refs[i];
        vector<int>::iterator it = find(ram.begin(), ram.end(), page);
        if (it != ram.end()) { ram.erase(it); ram.push_back(page); continue; }
        faults++;
        if ((int)ram.size() == frames) ram.pop_back();   // newest leaves
        ram.push_back(page);
    }
    return faults;
}

// LFU - throw out whoever has been used the FEWEST times overall.
long long runLFU(vector<int> &refs, int frames) {
    vector<int> ram;
    map<int, long long> count;
    long long faults = 0;
    for (int i = 0; i < (int)refs.size(); i++) {
        int page = refs[i];
        count[page]++;
        if (find(ram.begin(), ram.end(), page) != ram.end()) continue;
        faults++;
        if ((int)ram.size() == frames) {
            int victim = ram[0];
            for (int j = 0; j < (int)ram.size(); j++)
                if (count[ram[j]] < count[victim]) victim = ram[j];
            ram.erase(find(ram.begin(), ram.end(), victim));
        }
        ram.push_back(page);
    }
    return faults;
}

// RANDOM - throw out any page at random. This is the FLOOR: any policy that
// cannot beat random guessing is broken.
long long runRandom(vector<int> &refs, int frames) {
    vector<int> ram;
    long long faults = 0;
    mt19937 rng(11);
    for (int i = 0; i < (int)refs.size(); i++) {
        int page = refs[i];
        if (find(ram.begin(), ram.end(), page) != ram.end()) continue;
        faults++;
        if ((int)ram.size() == frames) {
            uniform_int_distribution<int> pick(0, (int)ram.size() - 1);
            ram.erase(ram.begin() + pick(rng));
        }
        ram.push_back(page);
    }
    return faults;
}

// CLOCK (Second Chance) - what real operating systems actually use.
// True LRU is too costly in hardware, so instead each page gets ONE bit.
// Bit = 1 means "used since we last looked". A hand goes round the frames:
//   bit 1 -> clear it to 0 and give the page a second chance, move on
//   bit 0 -> evict this one
long long runCLOCK(vector<int> &refs, int frames) {
    vector<int> ram;      // the pages, arranged in a circle
    vector<int> refbit;   // one bit per page
    int hand = 0;
    long long faults = 0;
    for (int i = 0; i < (int)refs.size(); i++) {
        int page = refs[i];
        int slot = -1;
        for (int j = 0; j < (int)ram.size(); j++) if (ram[j] == page) { slot = j; break; }
        if (slot != -1) { refbit[slot] = 1; continue; }        // HIT: just set the bit
        faults++;
        if ((int)ram.size() < frames) { ram.push_back(page); refbit.push_back(1); continue; }
        while (refbit[hand] == 1) {                            // second chances
            refbit[hand] = 0;
            hand = (hand + 1) % frames;
        }
        ram[hand] = page;                                      // evict here
        refbit[hand] = 1;
        hand = (hand + 1) % frames;
    }
    return faults;
}

// BELADY - the perfect policy. Throws out the page needed FURTHEST in future.
// Impossible in real life (needs to see the future) - we use it as the ceiling.
long long runBelady(vector<int> &refs, int frames) {
    vector<int> ram;
    long long faults = 0;
    map<int, deque<int> > nextUse;                       // page -> list of future positions
    for (int i = 0; i < (int)refs.size(); i++) nextUse[refs[i]].push_back(i);

    for (int i = 0; i < (int)refs.size(); i++) {
        int page = refs[i];
        if (!nextUse[page].empty() && nextUse[page].front() == i) nextUse[page].pop_front();
        if (find(ram.begin(), ram.end(), page) != ram.end()) continue;
        faults++;
        if ((int)ram.size() == frames) {
            int victim = ram[0];
            long long farthest = -1;
            for (int j = 0; j < (int)ram.size(); j++) {
                long long nxt = nextUse[ram[j]].empty() ? INF : nextUse[ram[j]].front();
                if (nxt > farthest) { farthest = nxt; victim = ram[j]; }
            }
            ram.erase(find(ram.begin(), ram.end(), victim));
        }
        ram.push_back(page);
    }
    return faults;
}

// ------------------------------------------------------------
// SECTION 3: PAQ - OUR ALGORITHM
//
// IDEA: don't pick between LRU and MRU. Instead LEARN what a
// "safe to throw out" page looks like.
//
// Every page in RAM gets described by 2 numbers:
//    recency group  = how long ago was it used   (0 to 7)
//    frequency group= how many times was it used (0 to 7)
// That gives 8 x 8 = 64 "types" of page.
// We keep ONE score per type in a table called Q.
//    score near +1  ->  this type is safe to throw out
//    score near -1  ->  throwing out this type was a mistake
//
// GHOST LIST: when we throw a page out, we write a receipt
// (page number, its type, the time). We do NOT keep the page data.
// If that page comes back quickly  -> we were wrong -> score goes down.
// If it stays away past window W   -> we were right -> score goes up.
// ------------------------------------------------------------

struct GhostEntry {
    int page;             // which page we threw out
    int rGroup, fGroup;   // which "type" it was when we threw it
    long long timeOut;    // when we threw it out
};

long long runPAQ(vector<int> &refs, int frames, bool keepLearning,
                 double qOut[R_BUCKET][F_BUCKET] = NULL) {
    double Q[R_BUCKET][F_BUCKET];
    for (int i = 0; i < R_BUCKET; i++)
        for (int j = 0; j < F_BUCKET; j++) Q[i][j] = 0.0;

    map<int, long long> lastUsed;   // page -> time it was last touched
    map<int, long long> useCount;   // page -> how many times touched
    deque<GhostEntry> ghost;        // the receipts

    double alpha = 0.1;             // learning rate: how fast scores move
    double eps   = 0.10;            // chance of trying a random eviction
    long long W  = 4LL * frames;    // "came back quickly" window
    size_t ghostCap = 8 * frames;

    mt19937 rng(7);
    uniform_real_distribution<double> coin(0.0, 1.0);

    vector<int> ram;
    long long faults = 0;

    for (long long t = 0; t < (long long)refs.size(); t++) {
        int page = refs[t];

        // --- STEP 1: throw away expired receipts. Those were GOOD evictions.
        while (!ghost.empty() && t - ghost.front().timeOut > W) {
            GhostEntry g = ghost.front(); ghost.pop_front();
            if (keepLearning) Q[g.rGroup][g.fGroup] += alpha * (1.0 - Q[g.rGroup][g.fGroup]);
        }
        while (ghost.size() > ghostCap) {
            GhostEntry g = ghost.front(); ghost.pop_front();
            if (keepLearning) Q[g.rGroup][g.fGroup] += alpha * (1.0 - Q[g.rGroup][g.fGroup]);
        }

        bool hit = (find(ram.begin(), ram.end(), page) != ram.end());

        // --- STEP 2: if this is a fault, check if WE caused it.
        if (!hit) {
            for (deque<GhostEntry>::iterator g = ghost.begin(); g != ghost.end(); ++g) {
                if (g->page == page) {
                    if (t - g->timeOut <= W && keepLearning)     // came back fast = our mistake
                        Q[g->rGroup][g->fGroup] += alpha * (-1.0 - Q[g->rGroup][g->fGroup]);
                    ghost.erase(g);
                    break;
                }
            }
        }

        lastUsed[page] = t;
        useCount[page]++;
        if (hit) continue;
        faults++;

        // --- STEP 3: if RAM is full, choose a victim using the learned table.
        if ((int)ram.size() == frames) {
            int victim = ram[0];
            if (keepLearning && coin(rng) < eps) {
                uniform_int_distribution<int> pickAny(0, (int)ram.size() - 1);
                victim = ram[pickAny(rng)];                    // explore
            } else {
                double best = -1e18;
                for (int j = 0; j < (int)ram.size(); j++) {
                    int p = ram[j];
                    long long age = max(1LL, t - lastUsed[p]);
                    int r = min(R_BUCKET - 1, (int)floor(log2((double)age)));
                    int f = min(F_BUCKET - 1, (int)floor(log2((double)max(1LL, useCount[p]))));
                    if (Q[r][f] > best) { best = Q[r][f]; victim = p; }
                }
            }
            // write the receipt before the page disappears
            long long age = max(1LL, t - lastUsed[victim]);
            GhostEntry g;
            g.page = victim;
            g.rGroup = min(R_BUCKET - 1, (int)floor(log2((double)age)));
            g.fGroup = min(F_BUCKET - 1, (int)floor(log2((double)max(1LL, useCount[victim]))));
            g.timeOut = t;
            ghost.push_back(g);

            ram.erase(find(ram.begin(), ram.end(), victim));
        }
        ram.push_back(page);
    }

    if (qOut != NULL)
        for (int i = 0; i < R_BUCKET; i++)
            for (int j = 0; j < F_BUCKET; j++) qOut[i][j] = Q[i][j];

    return faults;
}

// ------------------------------------------------------------
// SECTION 4: the score we report
// gapClosed = how much of the room between LRU and perfect we captured
// ------------------------------------------------------------
double gapClosed(long long lru, long long ours, long long best) {
    if (lru == best) return 0.0;
    return 100.0 * (double)(lru - ours) / (double)(lru - best);
}

// ------------------------------------------------------------
// SECTION 5: CORRECTNESS CHECKS
// If these fail, the simulator is broken and no result can be trusted.
// ------------------------------------------------------------
void runChecks() {
    cout << "=========== CORRECTNESS CHECKS ===========\n";
    mt19937 rng(3);
    vector<int> z = makeZipf(30, 5000, 0.9, rng);

    long long best = runBelady(z, 8);
    bool ok1 = (runFIFO(z,8) >= best && runLRU(z,8) >= best &&
                runLFU(z,8) >= best && runPAQ(z,8,true) >= best);
    cout << (ok1 ? "[PASS]" : "[FAIL]") << " Nobody beats Belady (it is the true ceiling)\n";

    vector<int> anom;
    int seq[] = {1,2,3,4,1,2,5,1,2,3,4,5};
    for (int i = 0; i < 12; i++) anom.push_back(seq[i]);
    long long f3 = runFIFO(anom, 3), f4 = runFIFO(anom, 4);
    cout << ((f4 > f3) ? "[PASS]" : "[FAIL]")
         << " Belady's Anomaly: FIFO with 3 frames = " << f3
         << ", with 4 frames = " << f4 << " (more RAM, MORE faults!)\n";

    vector<int> loop = makeLoop(5, 4000);
    long long lruLoop = runLRU(loop, 4);
    cout << ((lruLoop >= 3990) ? "[PASS]" : "[FAIL]")
         << " LRU on a loop = " << lruLoop << "/4000 faults (0% hit rate)\n";

    mt19937 rng2(5);
    vector<int> loc = makeZipf(25, 8000, 1.1, rng2);
    cout << ((runLRU(loc,6) < runFIFO(loc,6)) ? "[PASS]" : "[FAIL]")
         << " LRU (" << runLRU(loc,6) << ") beats FIFO (" << runFIFO(loc,6) << ") on normal work\n\n";
}

// ------------------------------------------------------------
// SECTION 6: MAIN - run everything and print the results
// ------------------------------------------------------------
int main() {
    cout << fixed << setprecision(1);
    runChecks();

    mt19937 rng(42);
    vector<int> wScan  = makeScan(20000);
    vector<int> wLoop  = makeLoop(FRAMES + 1, 20000);
    vector<int> wZipf  = makeZipf(60, 20000, 0.9, rng);
    vector<int> wPhase = makePhaseShift(6000, 3, rng);

    string names[4] = {"Scan", "Loop", "Zipf", "PhaseShift"};
    vector<int> *wl[4] = {&wScan, &wLoop, &wZipf, &wPhase};

    cout << "=========== SCOREBOARD (RAM = " << FRAMES << " frames) ===========\n";
    cout << left << setw(12) << "Workload" << right
         << setw(8) << "Random" << setw(8) << "FIFO" << setw(8) << "CLOCK"
         << setw(8) << "LRU" << setw(8) << "MRU" << setw(8) << "LFU"
         << setw(8) << "PAQ" << setw(8) << "Belady" << setw(11) << "PAQ gap%" << "\n";

    for (int i = 0; i < 4; i++) {
        long long rnd  = runRandom(*wl[i], FRAMES);
        long long fifo = runFIFO(*wl[i], FRAMES);
        long long clk  = runCLOCK(*wl[i], FRAMES);
        long long lru  = runLRU(*wl[i], FRAMES);
        long long mru  = runMRU(*wl[i], FRAMES);
        long long lfu  = runLFU(*wl[i], FRAMES);
        long long paq  = runPAQ(*wl[i], FRAMES, true);
        long long bel  = runBelady(*wl[i], FRAMES);
        cout << left << setw(12) << names[i] << right
             << setw(8) << rnd << setw(8) << fifo << setw(8) << clk
             << setw(8) << lru << setw(8) << mru << setw(8) << lfu
             << setw(8) << paq << setw(8) << bel
             << setw(10) << gapClosed(lru, paq, bel) << "%\n";
    }

    // Does the conclusion survive when RAM gets bigger? Test 4, 8, 16, 32 frames.
    //
    // IMPORTANT: the Loop and PhaseShift workloads must be REBUILT for each RAM
    // size. A loop is only adversarial when it is slightly LARGER than RAM. If we
    // reused a fixed 9-page loop at 32 frames the whole loop would sit in RAM, LRU
    // would already be perfect, and we would be measuring nothing.
    cout << "\n=========== FRAME-SIZE SWEEP (PAQ gap closed %) ===========\n";
    cout << "(Loop and PhaseShift are regenerated per size so the loop stays just larger than RAM)\n";
    cout << left << setw(12) << "Workload" << right
         << setw(11) << "4 frames" << setw(11) << "8 frames"
         << setw(11) << "16 frames" << setw(11) << "32 frames" << "\n";
    int sizes[4] = {4, 8, 16, 32};

    for (int i = 0; i < 4; i++) {
        cout << left << setw(12) << names[i] << right;
        for (int s = 0; s < 4; s++) {
            int F = sizes[s];
            vector<int> trace;
            mt19937 r2(42);
            if (names[i] == "Scan")            trace = makeScan(20000);
            else if (names[i] == "Loop")       trace = makeLoop(F + 1, 20000);
            else if (names[i] == "Zipf")       trace = makeZipf(60, 20000, 0.9, r2);
            else                               trace = makePhaseShiftFor(F, 6000, 3, r2);

            long long lru = runLRU(trace, F);
            long long paq = runPAQ(trace, F, true);
            long long bel = runBelady(trace, F);
            cout << setw(10) << gapClosed(lru, paq, bel) << "%";
        }
        cout << "\n";
    }

    // THE KEY EXPERIMENT: does letting PAQ keep learning during the test matter?
    cout << "\n=========== ONLINE vs FROZEN (PhaseShift) ===========\n";
    long long lruP = runLRU(wPhase, FRAMES);
    long long belP = runBelady(wPhase, FRAMES);
    long long onl  = runPAQ(wPhase, FRAMES, true);    // keeps learning
    long long frz  = runPAQ(wPhase, FRAMES, false);   // learning switched off
    cout << "LRU            " << setw(7) << lruP << " faults\n";
    cout << "Belady         " << setw(7) << belP << " faults  (perfect)\n";
    cout << "PAQ frozen     " << setw(7) << frz  << " faults  -> " << gapClosed(lruP,frz,belP) << "% of gap closed\n";
    cout << "PAQ online     " << setw(7) << onl  << " faults  -> " << gapClosed(lruP,onl,belP) << "% of gap closed\n";

    // THE LEARNED TABLE: proof the agent discovered a rule by itself
    cout << "\n=========== WHAT PAQ LEARNED (Zipf workload) ===========\n";
    cout << "Rows = recency group (0 = just used, 7 = long ago)\n";
    cout << "Cols = frequency group (0 = used once, 7 = used very often)\n";
    cout << "Value: +1 = safe to evict, -1 = do NOT evict this type\n\n";
    double Q[R_BUCKET][F_BUCKET];
    runPAQ(wZipf, FRAMES, true, Q);
    cout << "        ";
    for (int j = 0; j < F_BUCKET; j++) cout << setw(7) << ("f" + to_string(j));
    cout << "\n";
    for (int i = 0; i < R_BUCKET; i++) {
        cout << "  r" << i << "   ";
        for (int j = 0; j < F_BUCKET; j++) cout << setw(7) << setprecision(2) << Q[i][j];
        cout << "\n";
    }
    cout << setprecision(1) << "\nDone.\n";
    return 0;
}
