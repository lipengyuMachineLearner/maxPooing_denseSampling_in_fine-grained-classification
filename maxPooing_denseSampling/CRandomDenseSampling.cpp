#pragma once

#include "CRandomDenseSampling.h"
#include <time.h>
#include <fstream>
using namespace std;
int random(int min, int max);
void checkPath(string path);
float gaussrand(float ave, float var);

void CRandomDenseSampling::RandomDenseSampling(string segmentPath)
{
	cout << "##################" << endl << "RandomDenseSampling... ";
	logfile_ << "##################" << endl << "RandomDenseSampling... ";
	srand((int)time(0));
	
	for (int i_file = 0; i_file < num_img_; i_file++)
	{
		IplImage *img = cvLoadImage(imgFiles_[i_file].c_str(), CV_LOAD_IMAGE_COLOR);
		int width_img = img->width;
		int height_img = img->height;
		int nchannel_img = img->nChannels;
		int step_img = img->widthStep;
		
		int label = label_[i_file];
		string::size_type pos1 = imgFiles_[i_file].rfind("\\");
		string::size_type pos2 = imgFiles_[i_file].rfind("\\", pos1 - 1);
		string name_file = imgFiles_[i_file].substr(pos2 + 1, pos1 - pos2 - 1);

		pos2 = imgFiles_[i_file].rfind(".");
		string name_src = imgFiles_[i_file].substr(pos1 + 1, pos2 - pos1 - 1);

		
		
		string savePath = saveRoot_ + "\\" + name_file;
		checkPath(savePath);
		Mat mask;
		if (segmentPath == "")
			mask = Mat::ones(height_img, width_img, CV_8U);
		else
			loadMask(segmentPath, name_file + "\\" + name_src, mask, width_img, height_img);

		if (i_file % 1 == 0)
		{
			cout << "RandomDenseSampling " << i_file << " vs " << num_img_ << ", " << imgFiles_[i_file] << endl;
			logfile_ << "RandomDenseSampling " << i_file << " vs " << num_img_ << ", " << imgFiles_[i_file] << endl;
		}

		for (int i_scalar = 0; i_scalar < num_scalar_; i_scalar++)
		{
			int width_scalar = scalar_[i_scalar];
			int height_scalar = scalar_[i_scalar];
			int iter_threshold = 100 * num_kernelInScalar_[i_scalar];
			for (int i_kernel = 0; i_kernel < num_kernelInScalar_[i_scalar]; )
			{
				if (iter_threshold == 0)
				{
					cout << "do not get enough patches in " << imgFiles_[i_file] << " " << num_kernelInScalar_[i_scalar] << " vs " << i_kernel << endl;
					logfile_ << "do not get enough patches in " << imgFiles_[i_file] << " " << num_kernelInScalar_[i_scalar] << " vs " << i_kernel << endl;
					break;
				}

				int x = random(width_scalar, width_img - width_scalar);//gaussrand(width_img / 2.0, (width_img*0.75));// 
				int y = random(height_scalar, height_img - height_scalar);//gaussrand(height_img / 2.0, (height_img*0.75));;// 
				Point center(x, y);

				if (mask.at<unsigned char >(y, x) == 255)
				{
					i_kernel++;
				}
				else
				{
					iter_threshold--;
					//cout << iter_threshold << endl;
					continue;
				}
				

				CSamplingKernel kernel(width_scalar, height_scalar);
				kernel.setIndexAndLabel(label_[i_file], index_);
				kernel.setSrcName(name_src);
				kernel.setData(img, center);
				
				char buf[100];
				sprintf(buf, "%d", index_);
				string tmp = buf;
				string saveName = savePath + "\\" + name_src + "_" + tmp + ".bin";
				kernel.save(saveName);

				index_++;
			}
		}
	}

	cout << "##################" << endl;
	logfile_ << "##################" << endl;
}

bool CRandomDenseSampling::loadMask(string segmentPath, string name, Mat &mask, int width, int height)
{
	ifstream infile;
	string segmentName = segmentPath + '\\' + name + ".txt";
	infile.open(segmentName);
	mask = Mat::zeros(height, width, CV_8U);
	int data;
	for (int h = 0; h < height; h++)
	{
		for (int w = 0; w < width; w++)
		{
			infile >> data;
			mask.at<unsigned char>(h, w) = data*255;
		}
	}

	//imshow("mask",mask);
	//cvWaitKey(0);

	return true;
}
void CRandomDenseSampling::bin2img(string savePath)
{
	cout << "##################" << endl << "bin2imging... ";
	logfile_ << "##################" << endl << "bin2imging... ";
	checkPath(savePath);
	CSamplingKernel kernel;
	for (int i_file = 0; i_file < num_img_; i_file++)
	{
		if (i_file % 100 == 0)
		{
			cout << "bin2img " << i_file << " vs " << num_img_ << endl;
			logfile_ << "bin2img " << i_file << " vs " << num_img_ << endl;
		}

		kernel.load(imgFiles_[i_file]);

		string::size_type pos1 = imgFiles_[i_file].rfind("\\");
		string::size_type pos2 = imgFiles_[i_file].rfind(".");
		string saveName = imgFiles_[i_file].substr(pos1 + 1, pos2 - pos1 - 1);
		
		pos2 = imgFiles_[i_file].rfind("\\", pos1 - 1);
		string fileName = imgFiles_[i_file].substr(pos2 + 1, pos1 - pos2 - 1);

		checkPath(savePath + "\\" + fileName);
		saveName = savePath + "\\" + fileName + "\\" + saveName + ".jpg";
		kernel.saveAsImg(saveName);
	}

	cout << "##################" << endl;
	logfile_ << "##################" << endl;
}

string CRandomDenseSampling::cof_ = "1";
CRandomDenseSampling::CRandomDenseSampling(int num_img, const  string saveRoot, vector<string> imgFiles,
	vector<int> label, const int *scalar, const int *num_kernelInScalar, const int num_scalar)
{
	num_img_ = num_img;
	saveRoot_ = saveRoot;
	imgFiles_ = imgFiles;
	label_ = label;
	num_scalar_ = num_scalar;
	scalar_ = new int[num_scalar];
	_memccpy(scalar_, scalar, num_scalar*sizeof(int), num_scalar*sizeof(int));

	num_kernelInScalar_ = new int[num_scalar];
	_memccpy(num_kernelInScalar_, num_kernelInScalar, num_scalar*sizeof(int), num_scalar*sizeof(int));

	index_ = 0;

	logfile_.open("logfile_CRandomDenseSampling_" + cof_ + ".txt");
	cof_ += "1";
}

CRandomDenseSampling::~CRandomDenseSampling()
{
	delete []scalar_;
	delete[]num_kernelInScalar_;
}