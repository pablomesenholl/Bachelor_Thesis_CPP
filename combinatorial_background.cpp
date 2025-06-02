#include "Pythia8/Pythia.h"
#include "TFile.h"
#include "TTree.h"
#include "TLorentzVector.h"
#include "Math/Minimizer.h"
#include "Math/Factory.h"
#include "Math/Functor.h"
#include <random>
#include <iostream>
#include <array>
#include <algorithm>


// Define simplified geometric vertex chi2 minimization using ROOT::Math::Minimizer

struct Track {
    TVector3 origin;
    TVector3 direction;
};

// Chi2 function: sum of squared perpendicular distances to each track
class VertexChi2Functor {
public:
    VertexChi2Functor(const std::vector<Track>& tracks, double sigma)
      : fTracks(tracks), fSigma2(sigma * sigma) {}

    double operator()(const double* x) const {
        TVector3 V(x[0], x[1], x[2]);
        double chi2 = 0.0;
        for (const auto& t : fTracks) {
            TVector3 diff = V - t.origin;
            TVector3 perp = diff - diff.Dot(t.direction) * t.direction;
            chi2 += perp.Mag2() / fSigma2;
        }
        return chi2;
    }

private:
    const std::vector<Track>& fTracks;
    double fSigma2;
};

// Function to perform simplified vertex fit
bool FitVertex(const std::vector<Track>& tracks, double sigma, TVector3& fittedVertex, Float_t& chi2Out) {
    if (tracks.size() < 2) return false;

    // --- build a “composite” seed from the track origins, average starting point of tracks ---
    TVector3 seed(0,0,0);
    for (auto& t : tracks) seed += t.origin;
    seed *= (1.0 / tracks.size());

    ROOT::Math::Functor functor(VertexChi2Functor(tracks, sigma), 3);
    auto minimizer = std::unique_ptr<ROOT::Math::Minimizer>(
      ROOT::Math::Factory::CreateMinimizer("Minuit2", "Migrad"));

    minimizer->SetFunction(functor);
    minimizer->SetMaxFunctionCalls(200);
    minimizer->SetMaxIterations(500);
    minimizer->SetTolerance(1e-4);

    // Set initial guess (e.g. origin)
    minimizer->SetVariable(0, "x", seed.X(), 0.1);
    minimizer->SetVariable(1, "y", seed.Y(), 0.1);
    minimizer->SetVariable(2, "z", seed.Z(), 0.1);

    bool success = minimizer->Minimize();
    if (!success) return false;

    const double* xs = minimizer->X();
    fittedVertex.SetXYZ(xs[0], xs[1], xs[2]);
    chi2Out = minimizer->MinValue();
    return true;
}

// Compute average production vertex of a composite candidate
inline TVector3 CompositeVertex(const std::vector<Pythia8::Particle>& parts) {
    TVector3 sum(0,0,0);
    for (const auto& p : parts) {
        sum += TVector3(p.xProd(), p.yProd(), p.zProd());
    }
    return sum * (1.0 / parts.size());
}
inline double VertexDistance(const TVector3& a, const TVector3& b) { return (a-b).Mag(); }

// Define your distance threshold (in mm) for proximity filtering
const double kProximityThreshold = 0.1;
const size_t kMaxKstarCands = 50;  // cap Kπ pairs to top 50
const size_t kNearestM      = 5;   // only use 5 nearest neighbours for triplets

// Helper to compute 3D distance between two production vertices
inline double VertexDistance(const Pythia8::Particle& a, const Pythia8::Particle& b) {
    TVector3 va(a.xProd(), a.yProd(), a.zProd());
    TVector3 vb(b.xProd(), b.yProd(), b.zProd());
    return (va - vb).Mag();
}


using namespace Pythia8;

int main(int argc, char* argv[]) {
    int nEvents = 50000;
    if (argc > 1) nEvents = atoi(argv[1]);

    // Configure Pythia for pp collisions @ 13 TeV
    Pythia pythia;
    pythia.readString("Beams:idA = 2212");           // proton
    pythia.readString("Beams:idB = 2212");
    pythia.readString("Beams:eCM = 13000.");            // e.g. 13 TeV
    pythia.readString("Random:seed = 22");           // set seed
    pythia.readString("PhaseSpace:pTHatMin = 5");
    pythia.readString("HardQCD:all = off");
    pythia.readString("HardQCD:gg2ccbar   = on");
    pythia.readString("HardQCD:qqbar2ccbar= on");
    pythia.readString("HardQCD:gg2bbbar   = on");
    pythia.readString("HardQCD:qqbar2bbbar= on");
    pythia.readString("211:mayDecay = off");        // pions
    pythia.readString("-211:mayDecay = off");
    pythia.readString("321:mayDecay = off");        // kaons
    pythia.readString("-321:mayDecay = off");
    pythia.readString("13:mayDecay = off");        // muons
    pythia.readString("-13:mayDecay = off");


    // Vertex smearing for primary vertex (Gaussian)
    pythia.readString("Beams:allowVertexSpread = on");
    pythia.readString("Beams:sigmaVertexX = 0.01");         // mm (PV resolution)
    pythia.readString("Beams:sigmaVertexY = 0.01");         // mm
    pythia.readString("Beams:sigmaVertexZ = 0.025");         // mm
    pythia.readString("ParticleDecays:limitTau0 = on"); // limit very short lifetimes


    pythia.init();


    // Prepare output ROOT file & TTree
    TFile outFile("Combinatorial_Background_smeared.root", "RECREATE");
    TTree tree("Events", "Random combinatorial background");

    Float_t PVx, PVy, PVz;
    Float_t PVxErr = 0.01, PVyErr = 0.01, PVzErr = 0.025;
    // Kinematics placeholders
    Float_t kst_pt, kst_eta, kst_phi;
    Float_t tau1_pt, tau1_eta, tau1_phi;
    Float_t tau3_pt, tau3_eta, tau3_phi;
    Float_t m_tau1, m_tau3, m_kst;
    Float_t SVx, SVy, SVz;
    Float_t SVxErr = 0.005, SVyErr = 0.005, SVzErr = 0.01; // example SV smearing
    Float_t vertexChi2;

    // Fill TTree branches
    tree.Branch("PVx", &PVx, "PVx/F");
    tree.Branch("PVy", &PVy, "PVy/F");
    tree.Branch("PVz", &PVz, "PVz/F");
    tree.Branch("PVxErr", &PVxErr, "PVxErr/F");
    tree.Branch("PVyErr", &PVyErr, "PVyErr/F");
    tree.Branch("PVzErr", &PVzErr, "PVzErr/F");
    tree.Branch("SVx",       &SVx,       "SVx/F");
    tree.Branch("SVy",       &SVy,       "SVy/F");
    tree.Branch("SVz",       &SVz,       "SVz/F");
    tree.Branch("SVxErr",    &SVxErr,    "SVxErr/F");
    tree.Branch("SVyErr",    &SVyErr,    "SVyErr/F");
    tree.Branch("SVzErr",    &SVzErr,    "SVzErr/F");
    tree.Branch("vertexChi2", &vertexChi2, "vertexChi2/F");
    tree.Branch("kst_pt", &kst_pt, "kst_pt/F");
    tree.Branch("kst_eta", &kst_eta, "kst_eta/F");
    tree.Branch("kst_phi", &kst_phi, "kst_phi/F");
    tree.Branch("tau1_pt", &tau1_pt, "tau1_pt/F");
    tree.Branch("tau1_eta", &tau1_eta, "tau1_eta/F");
    tree.Branch("tau1_phi", &tau1_phi, "tau1_phi/F");
    tree.Branch("tau3_pt", &tau3_pt, "tau3_pt/F");
    tree.Branch("tau3_eta", &tau3_eta, "tau3_eta/F");
    tree.Branch("tau3_phi", &tau3_phi, "tau3_phi/F");
    tree.Branch("m_tau3", &m_tau3, "m_tau3/F");
    tree.Branch("m_tau1", &m_tau1, "m_tau1/F");
    tree.Branch("m_kst", &m_kst, "m_kst/F");

    // Random number generators for measurement smearing and uncertainties
    std::mt19937 rng(40);
    std::normal_distribution<double> smearSVxy(0.01);    // mm (SV spatial resolution)
    std::normal_distribution<double> smearSVz(0.01);      // mm

    std::vector<Particle>    storeK;      // K± for K*
    std::vector<Particle>    storePi;     // π± for K* and 3-prong
    std::vector<Particle>    tau1Cand;    // muons from τ→μνν
    std::vector<Particle>    prongs;      // pions from τ→3πν

    for (int iEvent = 0; iEvent < nEvents; ++iEvent) {
        if (iEvent % 100 == 0) {
            std::cout << "Event" << iEvent << " and " << nEvents << "\n";
        }
        if (!pythia.next()) continue;

        // Clear containers at start of each event
        storeK.clear();
        storePi.clear();
        tau1Cand.clear();
        prongs.clear();

        // Primary vertex (already smeared by Pythia)
        PVx = pythia.event[0].xProd();
        PVy = pythia.event[0].yProd();
        PVz = pythia.event[0].zProd();

        for (int i = 0; i < pythia.event.size(); ++i) {
            const Particle& p = pythia.event[i];
            if (!p.isFinal()) continue;
            // pick out all relevant particles and combine them randomly to fill Kstar Tau Tau kinematics here
            int absId = std::abs(p.id());
            // Collect K±
            if (absId == 321) storeK.push_back(p);
            // Collect π±
            if (absId == 211) storePi.push_back(p);
            // Collect μ±
            if (absId == 13)  tau1Cand.push_back(p);
            // Also use π± list for 3-prong: prongs
            if (absId == 211) prongs.push_back(p);
        }

        if (storeK.size() < 1 || storePi.size() < 4 || tau1Cand.size() < 1) {continue;}

        //std::cout << "Number of Kaons per event: " << storeK.size() << std::endl;
        //std::cout << "Number of Pions per event: " << storePi.size() << std::endl;
        //std::cout << "Number of Muons per event: " << tau1Cand.size() << std::endl;


        // Proximity-based K* candidates (but capped to the best kMaxKstarCands by distance)
        std::vector<std::pair<Particle, Particle>> kstarCands;
        for (auto& k : storeK) {
            for (auto& pi : storePi) {
                if (VertexDistance(k, pi) < kProximityThreshold)
                    kstarCands.emplace_back(k, pi);
            }
        }
        if (kstarCands.empty()) continue;

        // sort by their K–π distance and keep only the top kMaxKstarCands
        std::sort(kstarCands.begin(), kstarCands.end(),
          [&](auto &a, auto &b){
            return VertexDistance(a.first, a.second)
                 < VertexDistance(b.first, b.second);
          });
        if (kstarCands.size() > kMaxKstarCands)
            kstarCands.resize(kMaxKstarCands);
        //std::cout << "number of Kstar candidates: " << kstarCands.size() << std::endl;

        // Proximity-based 3-prong (tau3) candidates, but only among each pion's kNearestM neighbours
        std::vector<std::array<Particle,3>> tau3Cands;
        size_t nPi = prongs.size();
        if (nPi >= 3) {
            // 2a) build a list of kNearestM nearest neighbours for each pion
            std::vector<std::vector<size_t>> nearest(nPi);
            for (size_t i = 0; i < nPi; ++i) {
                std::vector<std::pair<double,size_t>> dists;
                dists.reserve(nPi-1);
                for (size_t j = 0; j < nPi; ++j) {
                    if (i == j) continue;
                    dists.emplace_back(VertexDistance(prongs[i], prongs[j]), j);
                }
                std::sort(dists.begin(), dists.end());
                // keep only the kNearestM closest
                for (size_t k = 0; k < std::min(kNearestM, dists.size()); ++k)
                    nearest[i].push_back(dists[k].second);
            }

            // 2b) form triplets among each pion + its nearest neighbours
            for (size_t i1 = 0; i1 < nPi; ++i1) {
                for (size_t idx2 = 0; idx2 < nearest[i1].size(); ++idx2) {
                    size_t i2 = nearest[i1][idx2];
                    for (size_t idx3 = idx2 + 1; idx3 < nearest[i1].size(); ++idx3) {
                        size_t i3 = nearest[i1][idx3];
                        tau3Cands.push_back({ prongs[i1],
                                              prongs[i2],
                                              prongs[i3] });
                    }
                }
            }
        }
        if (tau3Cands.empty()) continue;  // no nearby 3-pion triplets
        //std::cout << "number of tau3 candidates: " << tau3Cands.size() << std::endl;

        // Find best combination by minimal composite proximity
        double bestMetric=1e9;
        Particle bestK,bestPi,bestMu;
        std::array<Particle,3> bestTrip;
        for(auto& kp:kstarCands){
            TVector3 vKst=CompositeVertex({kp.first,kp.second});
            for(auto& trip:tau3Cands){
                TVector3 vT3=CompositeVertex({trip[0],trip[1],trip[2]});
                for(auto& mu:tau1Cand){
                    TVector3 vT1(mu.xProd(),mu.yProd(),mu.zProd());
                    // metric: max distance between composite vertices
                    double d1=VertexDistance(vKst,vT3), d2=VertexDistance(vKst,vT1), d3=VertexDistance(vT3,vT1);
                    double metric = std::max({d1,d2,d3});
                    if(metric<bestMetric){ bestMetric=metric; bestK=kp.first; bestPi=kp.second; bestMu=mu; bestTrip=trip; }
                }
            }
        }
        if(bestMetric>kProximityThreshold) continue;

        // Perform vertex fit on all 6 charged tracks (2 from K*, 1 muon, 3 pions), simplyfied version
        std::vector<Track> tracks;
        auto makeTrack = [&](const TLorentzVector& p4, const Particle& p) {
            // Use the particle's production vertex instead of the PV
            TVector3 origin(p.xProd(), p.yProd(), p.zProd());
            // (Optional) smear by detector resolution:
            origin += TVector3(smearSVxy(rng), smearSVxy(rng), smearSVz(rng));
            TVector3 dir = p4.Vect().Unit();
            tracks.push_back(Track{origin, dir});
        };
        TLorentzVector vK(bestK.px(), bestK.py(), bestK.pz(), bestK.e());
        TLorentzVector vPi(bestPi.px(),bestPi.py(),bestPi.pz(),bestPi.e());
        makeTrack(vK, bestK);
        makeTrack(vPi, bestPi);
        TLorentzVector vMu(bestMu.px(),bestMu.py(),bestMu.pz(),bestMu.e());
        makeTrack(vMu, bestMu);
        for (auto& p : bestTrip) {
            TLorentzVector v(p.px(),p.py(),p.pz(),p.e());
            makeTrack(v, p);
        }
        TLorentzVector vA(bestTrip[0].px(), bestTrip[0].py(), bestTrip[0].pz(), bestTrip[0].e());
        TLorentzVector vB(bestTrip[1].px(), bestTrip[1].py(), bestTrip[1].pz(), bestTrip[1].e());
        TLorentzVector vC(bestTrip[2].px(), bestTrip[2].py(), bestTrip[2].pz(), bestTrip[2].e());


        TVector3 vtx;
        float chi2;
        if (!FitVertex(tracks, 0.6, vtx, chi2)) continue;

        int ndof = static_cast<int>(tracks.size())*2 - 3;     // 2 constraints per track minus 3 free coords
        float chi2ndf = chi2 / ndof;
        SVx        = vtx.X();
        SVy        = vtx.Y();
        SVz        = vtx.Z();
        vertexChi2 = chi2ndf;

        // relative pt error for low pt << 100GeV
        double sigma_pt_rel = 0.007;

        TLorentzVector kstarVis = vK + vPi;
        TLorentzVector tau1Vis = vMu;
        TLorentzVector tau3Vis = vA + vB + vC;

        // store daughter kinematics and smear pt
        TLorentzVector kstar = vK + vPi;
        double sigma_pt_Kst = kstar.Pt() * sigma_pt_rel;
        std::normal_distribution<double> smearPt_Kst(kstar.Pt(), sigma_pt_Kst);
        kst_pt  = smearPt_Kst(rng); kst_eta = kstar.Eta(); kst_phi = kstar.Phi(); m_kst = kstar.M();
        TLorentzVector tau1 = vMu;
        double sigma_pt_tau1 = tau1.Pt() * sigma_pt_rel;
        std::normal_distribution<double> smearPt_tau1(tau1.Pt(), sigma_pt_tau1);
        tau1_pt  = smearPt_tau1(rng); tau1_eta = tau1.Eta(); tau1_phi = tau1.Phi(); m_tau1 = tau1.M();
        TLorentzVector t3 = vA+vB+vC;
        double sigma_pt_t3 = t3.Pt() * sigma_pt_rel;
        std::normal_distribution<double> smearPt_t3(t3.Pt(), sigma_pt_t3);
        tau3_pt  = smearPt_t3(rng); tau3_eta = t3.Eta(); tau3_phi = t3.Phi(); m_tau3 = t3.M();

        //std::cout << "kst_eta: " << kst_eta << std::endl;
        //std::cout << "tau1_eta: " << tau1_eta << std::endl;
        //std::cout << "tau3_eta: " << tau3_eta << std::endl;

        tree.Fill();
    }

    std::cout << "Entries in tree: " << tree.GetEntries() << "\n";

    // Finalize
    pythia.stat();
    outFile.Write();
    outFile.Close();

    return 0;
}