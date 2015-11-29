
#include <iostream>
#include <vector>
#include <stdlib.h>
#include <memory>
#include <direct.h>
#include "windows.h"
#include<time.h>
#include <io.h>
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

