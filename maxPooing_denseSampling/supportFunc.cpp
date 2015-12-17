#pragma once
#include <iostream>
#include <fstream>
#include <vector>
#include <stdlib.h>
#include <memory>
#include <direct.h>
#include "windows.h"
#include<time.h>
#include <io.h>
#include "opencv.h"
#include <hash_map>

using namespace std;

float gaussrand(float ave, float var)

{

	static float V1, V2, S;

	static int phase = 0;

	float X;



	if (phase == 0) {

		do {

			float U1 = (double)rand() / RAND_MAX;

			float U2 = (double)rand() / RAND_MAX;



			V1 = 2 * U1 - 1;

			V2 = 2 * U2 - 1;

			S = V1 * V1 + V2 * V2;

		} while (S >= 1 || S == 0);



		X = V1 * sqrt(-2 * log(S) / S);

	}
	else

		X = V2 * sqrt(-2 * log(S) / S);



	phase = 1 - phase;



	return X*var+ave;

}

int random(int min, int max)
{
	float ran = rand();
	ran = ran / RAND_MAX;
	ran = ran * (max - min) + min;

	return ran;
}

void checkPath(string path)
{
	WIN32_FIND_DATA wfd;
	bool rValue = false;
	HANDLE hFind = FindFirstFile((LPCWSTR)path.c_str(), &wfd);
	if ((hFind != INVALID_HANDLE_VALUE) && (wfd.dwFileAttributes && FILE_ATTRIBUTE_DIRECTORY))
	{
		rValue = true;
	}
	else
	{
		_mkdir(path.c_str());
	}
	FindClose(hFind);
}

int getFiles(string rootPath, string format, vector<string> &files)
{
	struct _finddata_t fa;
	long fHandle;

	if ((fHandle = _findfirst((rootPath + "\\" + "*").c_str(), &fa)) == -1L)
	{
		printf("there is no relative file in the rootPath\n");
		exit(0);
	}
	else
	{
		do
		{
			string name = fa.name;
			if (name == "." || name == "..")
				continue;
			if (name.find(format) != string::npos)
				files.push_back(rootPath + "\\" + name);
			else
				getFiles(rootPath + "\\" + name, format, files);
		} while (_findnext(fHandle, &fa) == 0);
		_findclose(fHandle);

		return files.size();
	}
}

void matSave_oneChannel(string fileName, Mat &data)
{
	ofstream outfileBin;
	outfileBin.open(fileName, ios::binary);
	int row = data.rows;
	int col = data.cols;
	
	outfileBin.write((char *)&row, sizeof(row));
	outfileBin.write((char *)&col, sizeof(col));

	for (int h = 0; h < row; h++)
	{
		for (int w = 0; w < col; w++)
		{
			float tmp = data.at<float>(h, w);
			outfileBin.write((char *)&tmp, sizeof(tmp));
		}
	}

	outfileBin.close();
}

void matLoad_oneChannel(string fileName, Mat &data)
{
	ifstream infileBin;
	infileBin.open(fileName, ios::binary);
	int row = 0;
	int col = 0;

	infileBin.read((char *)&row, sizeof(row));
	infileBin.read((char *)&col, sizeof(col));
	data.create(row, col, CV_32FC1);
	for (int h = 0; h < row; h++)
	{
		for (int w = 0; w < col; w++)
		{
			float tmp = 0;
			infileBin.read((char *)&tmp, sizeof(tmp));
			data.at<float>(h, w) = tmp;
		}
	}

	infileBin.close();
}

void mapSave(string fileName, hash_map<int, int> &feature2kernel)
{
	ofstream outfile;
	outfile.open(fileName);

	for (hash_map<int, int>::iterator iter = feature2kernel.begin(); iter != feature2kernel.end(); iter++)
	{
		outfile << iter->first << " " << iter->second << endl;
	}

	outfile.close();

}

void mapLoad(string fileName, hash_map<int, int> &feature2kernel)
{
	ifstream infile;
	infile.open(fileName);

	int key, value;
	while (infile >> key >> value)
		feature2kernel[key] = value;

	infile.close();
}