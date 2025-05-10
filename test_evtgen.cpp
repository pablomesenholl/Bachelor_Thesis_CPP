#include "EvtGen/EvtGen.hh"
#include "EvtGenBase/EvtRandomEngine.hh"
#include "EvtGenBase/EvtSimpleRandomEngine.hh"
#include "EvtGenExternal/EvtExternalGenList.hh"
#include "EvtGenModels/EvtAbsExternalGen.hh"
#include "EvtGenBase/EvtAbsRadCorr.hh"
#include "EvtGenBase/EvtDecayBase.hh"
#include <iostream>

int main() {
    std::string decayFile = "/home/pablo/projects/bachelor_thesis_cpp/B0_Kst_tautau.dec";
    std::string pdtFile   = "/home/pablo/evtgen-new-install/share/EvtGen/evt.pdl";

    EvtRandomEngine* myRandomEngine = new EvtSimpleRandomEngine();

    EvtExternalGenList genList;
    EvtAbsRadCorr* photosEngine = genList.getPhotosModel();
    std::list<EvtDecayBase*> extraModels = genList.getListOfModels();

    try {
        // The fourth parameter is the radiation correction engine (optional), so we pass nullptr
        EvtGen myGenerator(decayFile, pdtFile, myRandomEngine, photosEngine, &extraModels);
        std::cout << "✅ EvtGen initialized successfully.\n";
    } catch (const std::exception& e) {
        std::cerr << "❌ Exception while initializing EvtGen: " << e.what() << "\n";
        return 1;
    } catch (...) {
        std::cerr << "❌ Unknown error occurred while initializing EvtGen.\n";
        return 1;
    }

    delete myRandomEngine;
    return 0;
}
