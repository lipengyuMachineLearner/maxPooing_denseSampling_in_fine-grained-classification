#include <iostream>
#include <fstream>
#include <vector>
#include <time.h>
#include "config.h"
#include "CRandomDenseSampling.h"
#include "CConvMaxPooling.h"
#include "CimgConvolution.h"
void matLoad_oneChannel(string fileName, Mat &data);
using namespace std;

int getUCB200Config(string configName, vector<string> &fileName_vec, vector<int> &label_vec)
{
	fstream configFile;
	configFile.open(ROOT_PATH + "\\" + configName);

	string line;
	while (configFile >> line)
	{
		string name;
		int label;
		string::size_type pos = line.find(".");
		label = atoi(line.substr(0, pos).c_str());

		name = DATASET_PATH + "\\" + line;
		fileName_vec.push_back(name);
		label_vec.push_back(label);
	}

	return fileName_vec.size();

}

int getFiles(string rootPath, string format, vector<string> &files);
int main()
{
	
	
	cout << "starting..." << endl;
	float *ttts = new float[10000*200*40];
	//densely sampling
	/*vector<string> fileName_vec;
	vector<int> label_vec;
	int num_img = getUCB200Config("config_t.txt", fileName_vec, label_vec);
	CRandomDenseSampling sampling(num_img, SAVEROOT_KERNEL, fileName_vec,
	label_vec, SCALAR, SCALAR_RATIO, NUM_SCALAR);
	sampling.RandomDenseSampling(PATH_SEGMENT);*/


	//visualize the dense sampling kernel 
	//vector<string> kernelName_vec;
	//vector<int> kernelLabel_vec;
	//string savePath = "F:\\machineLearning\\code\\maxPooing_denseSampling_in_fine-grained-classification\\experiment\\CUB200\\kernel_img";
	//int num_kernel = getFiles(SAVEROOT_KERNEL + "\\001.Black_footed_Albatross", ".bin", kernelName_vec);
	//CRandomDenseSampling sampling2(num_kernel, SAVEROOT_KERNEL, kernelName_vec,
	//	kernelLabel_vec, SCALAR, SCALAR_RATIO, NUM_SCALAR);
	//sampling2.bin2img(savePath);

	//CovMaxPooling
	vector<string> kernelName_vec;
	int num_kernel = getFiles(SAVEROOT_KERNEL+"\\001.Black_footed_Albatross", ".bin", kernelName_vec);
	vector<string> fileName_vec;
	vector<int> label_vec;
	int num_img = getUCB200Config("config_t.txt", fileName_vec, label_vec);
	CConvMaxPooling conMaxPooling(num_kernel, num_img, fileName_vec, label_vec, kernelName_vec, CONVOLUTION_PATH, NUM_TARGETCLASS, TARGETCLASS);

	//clock_t t1 = clock();
	//conMaxPooling.ConvMaxPooling(CONV_STEP,SIGN_EXPAND_KERNEL);
	//clock_t t2 = clock();
	//cout << "Complete MaxPooling, Running time=" << t2 - t1 << endl;

	clock_t t1 = clock();
	float error = conMaxPooling.train();
	clock_t t2 = clock();
	cout << "training error = " << error << endl;
	cout << "Complete training, Running time=" << t2 - t1 << endl;

	/*Mat tmp;
	matLoad_oneChannel("feature.bin", tmp);
	ofstream outfile;
	outfile.open("feature.csv");

	for (int h = 0; h < tmp.rows; h++)
	{
		for (int w = 0; w < tmp.cols; w++)
		{
			cout << tmp.at<float>(h,w) << ",";
			outfile << tmp.at<float>(h, w) << ",";
		}
		outfile << endl;
		cout << endl;
	}*/
	//CimgConvolution tmp;
	//tmp.load("F:\\machineLearning\\code\\maxPooing_denseSampling_in_fine-grained-classification\\experiment\\CUB200\\convolution\\001.Black_footed_Albatross\\Black_footed_Albatross_0002_2293084168_0.bin");
	//tmp.showAsImg();

}

