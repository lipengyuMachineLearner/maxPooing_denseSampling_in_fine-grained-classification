#pragma once
#include "CSamplingKernel.h"
#include "fstream"
class CRandomDenseSampling
{
public:
	static string cof_;
	std::ofstream logfile_;

private:
	int num_img_;
	int num_scalar_;
	int num_allKernel_;

	vector<string> imgFiles_;
	vector<int> label_;
	int *scalar_;
	int *num_kernelInScalar_;
	string saveRoot_;

	int index_;
public:
	CRandomDenseSampling(int num_img, const  string saveRoot, vector<string> imgFiles,
		vector<int> label_, const int *scalar, const int *num_kernelInScalar, const int num_scalar);
	~CRandomDenseSampling();
	void RandomDenseSampling(string segmentPath);
	void bin2img(string savePath);

private:
	bool loadMask(string segmentPath, string name, Mat &mask, int width, int height);
};