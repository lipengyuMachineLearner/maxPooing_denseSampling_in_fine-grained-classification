#include "CConvMaxPooling.h"
#include "CimgConvolution.h"

void checkPath(string path);
void matSave_oneChannel(string fileName, Mat &data);
void matLoad_oneChannel(string fileName, Mat &data);
void mapSave(string fileName, hash_map<int,int> &feature2kernel);
void mapLoad(string fileName, hash_map<int, int> &feature2kernel);
string CConvMaxPooling::cof_ = "1";
CConvMaxPooling::CConvMaxPooling(int num_kernel, int num_img, vector<string> &img_files, vector<int> &label_vec, vector<string> &kernel_files, string convolutionSavePath, int num_target, const int *targetClass)
{
	num_kernel_ = num_kernel;
	num_img_ = num_img;
	img_files_ = img_files;
	label_vec_ = label_vec;
	kernel_files_ = kernel_files;
	convolutionSavePath_ = convolutionSavePath;

	feature_ = Mat::zeros(num_img_, num_kernel_, CV_32FC1);
	vote_feature_ = Mat::zeros(num_kernel_, 1, CV_32FC1);

	num_target_ = num_target;
	targetClass_ = new int[num_target_];
	memcpy(targetClass_, targetClass, sizeof(int)*num_target_);

	logfile_.open("CConvMaxPooling" + cof_ + ".txt");
	cof_ += "1";

	feature_ = Mat::zeros(num_img_, num_kernel_, CV_32FC1);
}

void CConvMaxPooling::ConvMaxPooling(int step_conv, bool sign_expand_kernel)
{
	
	for (int i_img = 0; i_img < num_img_; i_img++)
	{
		cout << "conv image No." << i_img << " vs " << num_img_ << " : " << img_files_[i_img] << endl;
		logfile_ << "conv image No." << i_img << " vs " << num_img_ << " : " << img_files_[i_img] << endl;

		IplImage *img = cvLoadImage(img_files_[i_img].c_str(), CV_LOAD_IMAGE_COLOR);
		int width_img = img->width;
		int height_img = img->height;
		int step_img = img->widthStep;
		
#pragma omp parallel for
		for (int i_kernel = 0; i_kernel < num_kernel_; i_kernel++)
		{
			CSamplingKernel kernel;
			kernel.load(kernel_files_[i_kernel]);
			int id_kernel = kernel.getID();
			cout << "\t conv kernel No." << i_kernel << " vs " << num_kernel_ << "; kernel_id=" << id_kernel << "; sign_expand_kernel=" << sign_expand_kernel << endl;
			logfile_ << "\t conv kernel No." << i_kernel << " vs " << num_kernel_ << "; kernel_id=" << id_kernel << "; sign_expand_kernel=" << sign_expand_kernel << endl;

			CimgConvolution imgConvolution;
			float &max = imgConvolution.maxValue_;
			Point &maxPosition = imgConvolution.maxPosition_;
			int &width_convolution = imgConvolution.width_;
			int &height_convolution = imgConvolution.height_;
			Mat &img_convolution = imgConvolution.data_;

			max = kernel.convolution(img, img_convolution, width_convolution, height_convolution, maxPosition, sign_expand_kernel, step_conv);
			feature_.at<float>(i_img, i_kernel) = max;
			if (feature2kernel_.find(i_kernel) == feature2kernel_.end())
				feature2kernel_[i_kernel] = id_kernel;

			string::size_type pos1 = img_files_[i_img].rfind("\\");
			string::size_type pos2 = img_files_[i_img].rfind(".");
			string saveName = img_files_[i_img].substr(pos1 + 1, pos2 - pos1 - 1);

			pos2 = img_files_[i_img].rfind("\\", pos1 - 1);
			string fileName = img_files_[i_img].substr(pos2 + 1, pos1 - pos2 - 1);

			char tmp[10];
			sprintf(tmp, "%d", i_kernel);
			checkPath(convolutionSavePath_ + "\\" + fileName);
			saveName = convolutionSavePath_ + "\\" + fileName + "\\" + saveName + "_" + tmp + ".bin";
			imgConvolution.save(saveName);



			/*imgConvolution.showAsImg();
			int width_kernel, height_kernel, channel_kernel;
			std::cout << max << std::endl;
			kernel.getParam(width_kernel, height_kernel);
			Mat img_kernel;
			kernel.getData(img_kernel);

			cvRectangle(img, cvPoint(maxPosition.x, maxPosition.y), cvPoint(maxPosition.x + width_kernel, maxPosition.y + height_kernel), cvScalar(0,0,255), 2);
			cvShowImage("imgimg", img);
			imshow("img_kernel", img_kernel);
			cvWaitKey(0);*/
		}
	}

	matSave_oneChannel("feature.bin", feature_);
	mapSave("feature2kernel.txt", feature2kernel_);
	
}

bool CConvMaxPooling::generateConvolutionImage(vector<Mat> &img, Mat &img_convolution, int width, int height)
{
	int num = img_convolution.cols;

	for (int n = 0; n < num; n++)
	{
		Mat imgtmp(height, width, CV_32FC1);
		for (int h = 0; h < height; h++)
		{
			for (int w = 0; w < width; w++)
			{
				imgtmp.at<float>(h, w) = img_convolution.at<float>(w + h*width, n);
			}
		}

		img.push_back(imgtmp);
	}
	return true;
}

bool CConvMaxPooling::getLabel(int targetLabel, float *num_eachClass)
{
	label_.create(num_img_, 1, CV_32FC1);
	for (int i_img = 0; i_img < num_img_; i_img++)
	{
		string::size_type pos1 = img_files_[i_img].rfind("\\");
		string::size_type pos2 = img_files_[i_img].rfind("\\",pos1-1);
		pos1 = img_files_[i_img].find(".");
		string labelStr = img_files_[i_img].substr(pos2 + 1, pos1 - pos2 - 1);
		int labelTmp = atoi(labelStr.c_str());
		if (labelTmp == targetLabel)
		{
			label_.at<float>(i_img, 0) = 1;
			num_eachClass[1] += 1;
		}
		else
		{
			label_.at<float>(i_img, 0) = 0;
			num_eachClass[0] += 1;
		}
	}

	return true;
}


float CConvMaxPooling::train()
{
	float result = 0;
	//ConvMaxPooling(step_conv, sign_expand_kernel);
	matLoad_oneChannel("feature.bin", feature_);
	mapLoad("feature2kernel.txt", feature2kernel_);
	float num_eachClass[2] = { 0, 0 };
	for (int target_i = 0; target_i < num_target_; target_i++)
	{
		memset(num_eachClass, 0, sizeof(float)*2);
		getLabel(targetClass_[target_i], num_eachClass);
		result = trainWithRandomForest(num_eachClass, 2, targetClass_[target_i]);
	}

	return result;
}

bool CConvMaxPooling::getFeatureImportant(const CvDTreeNode  *root)
{
	if (root == NULL)
		return true;

	CvDTreeSplit *split = root->split;
	if (split == NULL)
		return true;
	int feature_id = split->var_idx;
	vote_feature_.at<float>(feature_id, 0) += 1;
	getFeatureImportant(root->left);
	getFeatureImportant(root->right);

	return true;
}

float CConvMaxPooling::trainWithRandomForest(float *num_eachClass, int dim, int targetClass)
{ 
	float sum = 0;
	for (int i = 0; i < dim; i++)
		sum += num_eachClass[i];
	for (int i = 0; i < dim; i++)
		num_eachClass[i] = num_eachClass[i] / sum;

	CvDTreeParams  params = CvDTreeParams
		(
			100,
			1,
			0,
			true,
			2,
			1,
			true,
			true,
			num_eachClass
		);

	CvDTree *classifier = new CvDTree();
	Mat varType(feature_.cols + 1, 1, CV_8UC1);

	for (int i = 0; i < varType.rows; i++)
		varType.at<unsigned char>(i, 0) = CV_VAR_NUMERICAL;

	varType.at<unsigned char>(feature_.cols, 0) = CV_VAR_CATEGORICAL;

	classifier->train
		(
			feature_,
			CV_ROW_SAMPLE,
			label_,
			Mat(),
			Mat(),
			varType,
			Mat(),
			params
		);

	float error = 0;
	for (int i_img = 0; i_img < num_img_; i_img++)
	{
		Mat row = feature_.row(i_img);
		float labelPredict = classifier->predict(row)->value;
		float labelGroundtruth = label_.at<float>(i_img, 0);
	
		if (abs(labelPredict - labelGroundtruth) > 0.5)
			error++;

		//cout << labelPredict << " vs " << labelGroundtruth << endl;
	}
	
	char tmp[33];
	sprintf(tmp, "%d", targetClass);
	string t = tmp;
	classifier->save(("CDTree_" + t + ".xml").c_str());


	const CvDTreeNode *root = classifier->get_root();
	getFeatureImportant(root);

	for (int i = 0; i < num_kernel_; i++)
	{
		if (abs(vote_feature_.at<float>(i, 0)) > 0.5)
			cout << vote_feature_.at<float>(i, 0) << " : " << i << " vs " << feature2kernel_[i] << endl;
	}
	//delete classifier;
	return error / num_img_; 
}