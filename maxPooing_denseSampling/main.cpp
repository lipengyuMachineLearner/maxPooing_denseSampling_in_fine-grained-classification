#include <iostream>
#include <fstream>
#include <vector>
#include "config.h"
#include "CRandomDenseSampling.h"
#include "CConvMaxPooling.h"

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
	
	
	

	//densely sampling
	//vector<string> fileName_vec;
	//vector<int> label_vec;
	//int num_img = getUCB200Config("config_train.txt", fileName_vec, label_vec);
	//CRandomDenseSampling sampling(num_img, SAVEROOT_KERNEL, fileName_vec,
	//label_vec, SCALAR, SCALAR_RATIO, NUM_SCALAR);
	//sampling.RandomDenseSampling(PATH_SEGMENT);


	//visualize the dense sampling kernel 
	vector<string> kernelName_vec;
	vector<int> kernelLabel_vec;
	string savePath = "F:\\machineLearning\\code\\maxPooing_denseSampling_in_fine-grained-classification\\experiment\\CUB200\\kernel_img";
	int num_kernel = getFiles(SAVEROOT_KERNEL + "\\001.Black_footed_Albatross", ".bin", kernelName_vec);
	CRandomDenseSampling sampling2(num_kernel, SAVEROOT_KERNEL, kernelName_vec,
		kernelLabel_vec, SCALAR, SCALAR_RATIO, NUM_SCALAR);
	sampling2.bin2img(savePath);

	//CovMaxPooling
	/*vector<string> kernelName_vec;
	int num_kernel = getFiles(SAVEROOT_KERNEL + "\\001.Black_footed_Albatross", ".bin", kernelName_vec);
	vector<string> fileName_vec;
	vector<int> label_vec;
	int num_img = getUCB200Config("config_train.txt", fileName_vec, label_vec);
	CConvMaxPooling conMaxPooling(num_kernel, num_img, fileName_vec, label_vec, kernelName_vec);
	conMaxPooling.train(SIGN_EXPAND_KERNEL, CONV_STEP);*/
}