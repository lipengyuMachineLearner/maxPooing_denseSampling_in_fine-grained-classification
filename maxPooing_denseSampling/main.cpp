#include <iostream>
#include <fstream>
#include <vector>
#include "config.h"
#include "CRandomDenseSampling.h"

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
	vector<string> fileName_vec;
	vector<int> label_vec;
	//int num = getUCB200Config("config_train.txt", fileName_vec, label_vec);

	//CRandomDenseSampling sampling(num, SAVEROOT_KERNEL, fileName_vec,
	//label_vec, SCALAR, SCALAR_RATIO, NUM_SCALAR);
	//sampling.RandomDenseSampling(PATH_SEGMENT);


	//visualize the dense sampling kernel 
	fileName_vec.clear();
	label_vec.clear();
	string rootPath = "F:\\machineLearning\\code\\maxPooing_denseSampling_in_fine-grained-classification\\experiment\\CUB200\\kernel";
	string savePath = "F:\\machineLearning\\code\\maxPooing_denseSampling_in_fine-grained-classification\\experiment\\CUB200\\kernel_img";
	int num = getFiles(rootPath, ".bin", fileName_vec);
	CRandomDenseSampling sampling2(num, SAVEROOT_KERNEL, fileName_vec,
		label_vec, SCALAR, SCALAR_RATIO,NUM_SCALAR);
	sampling2.bin2img(savePath);
}