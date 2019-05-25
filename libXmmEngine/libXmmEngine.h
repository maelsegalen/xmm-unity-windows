#pragma once
// this is our interface to Unity :
#define EXPORT_API __declspec(dllexport)
extern "C" {

	//========================== INSTANCES MANAGEMENT ==========================//

	// "hhmm", "gmm", or JSON string model
	extern "C" __declspec(dllexport) int createModelInstance(const char *modelType);
	extern "C" __declspec(dllexport) int deleteModelInstance(int index);

	extern "C" __declspec(dllexport) void setCurrentModelInstance(int index);
	extern "C" __declspec(dllexport) int getCurrentModelInstance();
	extern "C" __declspec(dllexport) int getNbOfModelInstances();

	extern "C" __declspec(dllexport) int createSetInstance();
	extern "C" __declspec(dllexport) int deleteSetInstance(int index);

	extern "C" __declspec(dllexport) void setCurrentSetInstance(int index);
	extern "C" __declspec(dllexport) int getCurrentSetInstance();
	extern "C" __declspec(dllexport) int getNbOfSetInstances();

	extern "C" __declspec(dllexport) int trainModelFromSet();
	extern "C" __declspec(dllexport) int trained();

	//======================== TRAINING SETS INTERFACE =========================//

	// we determine the modality when first phrase is added to the empty set
	extern "C" __declspec(dllexport) void addPhraseFromData(const char *label, const char **colNames, float *phrase,
		int dimIn, int dimOut, int phraseSize);
	extern "C" __declspec(dllexport) void addPhrase(const char *sp);
	extern "C" __declspec(dllexport) const char *getPhrase(int index);
	extern "C" __declspec(dllexport) void removePhrase(int index);
	extern "C" __declspec(dllexport) void removePhrasesOfLabel(const char *label);

	extern "C" __declspec(dllexport) int getTrainingSetSize();
	extern "C" __declspec(dllexport) int getTrainingSetNbOfLabels();

	extern "C" __declspec(dllexport) const char **getTrainingSetLabels();
	extern "C" __declspec(dllexport) const char *getTrainingSet();
	extern "C" __declspec(dllexport) const char *getSubTrainingSet(const char *label);

	extern "C" __declspec(dllexport) void setTrainingSet(const char *sts);
	extern "C" __declspec(dllexport) void addTrainingSet(const char *sts);
	extern "C" __declspec(dllexport) void clearTrainingSet();

	//============================ MODELS INTERFACE ============================//

	// config parameters getters :
	extern "C" __declspec(dllexport) int getModelType();            // 0 : GMM, 1 : HHMM
	extern "C" __declspec(dllexport) int getBimodal();
	extern "C" __declspec(dllexport) int getNbOfModels();
	extern "C" __declspec(dllexport) const char **getModelLabels();
	extern "C" __declspec(dllexport) int getInputDimension();
	extern "C" __declspec(dllexport) int getOutputDimension();
	extern "C" __declspec(dllexport) int getGaussians();
	extern "C" __declspec(dllexport) float getRelativeRegularization();
	extern "C" __declspec(dllexport) float getAbsoluteRegularization();
	// see xmmGaussianDistribution.hpp :
	extern "C" __declspec(dllexport) int getCovarianceMode();       // 0 : full, 1 : diagonal
	extern "C" __declspec(dllexport) int getHierarchical();
	extern "C" __declspec(dllexport) int getStates();
	// see xmmHmmParameters.hpp :
	extern "C" __declspec(dllexport) int getTransitionMode();      // 0 : ergodic, 1 : leftright
	extern "C" __declspec(dllexport) int getRegressionEstimator(); // 0 : full, 1 : windowed, 2 : likeliest

	// config parameters setters
	extern "C" __declspec(dllexport) void setGaussians(int g);
	extern "C" __declspec(dllexport) void setRelativeRegularization(float relReg);
	extern "C" __declspec(dllexport) void setAbsoluteRegularization(float absReg);
	extern "C" __declspec(dllexport) void setCovarianceMode(int c);
	extern "C" __declspec(dllexport) void setHierarchical(int h);
	extern "C" __declspec(dllexport) void setStates(int s);
	extern "C" __declspec(dllexport) void setTransitionMode(int t);
	extern "C" __declspec(dllexport) void setRegressionEstimator(int r);

	// filtering related methods
	extern "C" __declspec(dllexport) int getLikelihoodWindow();
	extern "C" __declspec(dllexport) void setLikelihoodWindow(int w);

	// TODO : allow asynchronous training ?
	//int isTraining();
	//void cancelTraining();

	extern "C" __declspec(dllexport) const char *getModel();
	extern "C" __declspec(dllexport) void clearModel();

	extern "C" __declspec(dllexport) void filter(float *observation, int observationSize);

	extern "C" __declspec(dllexport) const char *getFilteringResults();  // JSON string (not very useful)
	extern "C" __declspec(dllexport) const char *getLikeliest();         // string label
	extern "C" __declspec(dllexport) float *getLikelihoods();      // we have "getNbOfModels()" likelihoods
	extern "C" __declspec(dllexport) float *getTimeProgressions(); // we have "getNbOfModels()" time progressions
	extern "C" __declspec(dllexport) float *getRegression();       // we have "getOutputDimension()" output values

	extern "C" __declspec(dllexport) void reset();
}
