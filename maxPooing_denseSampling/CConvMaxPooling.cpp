#include "CConvMaxPooling.h"
#include "CimgConvolution.h"

void checkPath(string path);
void matSave_oneChannel(string fileName, Mat &data);
void matLoad_oneChannel(string fileName, Mat &data);
void mapSave(string fileName, hash_map<int,int> &feature2kernel);

string CConvMaxPooling::cof_ = "1";
CConvMaxPooling::CConvMaxPooling(int num_kernel, int num_img, vector<string> &img_files, vector<int> &label_vec, vector<string> &kernel_files, string convolutionSavePath)
{
	num_kernel_ = num_kernel;
	num_img_ = num_img;
	img_files_ = img_files;
	label_vec_ = label_vec;
	kernel_files_ = kernel_files;
	convolutionSavePath_ = convolutionSavePath;

	feature_ = Mat::zeros(num_img_, num_kernel_, CV_32FC1);
	vote_feature_ = Mat::zeros(num_kernel_, 1, CV_32FC1);

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



			//imgConvolution.showAsImg();
			/*int width_kernel, height_kernel, channel_kernel;
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
	mapSave("feature2kernel.csv", feature2kernel_);
	
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

float CConvMaxPooling::train(bool sign_expand_kernel, int step_conv)
{
	float result = 0;
	ConvMaxPooling(step_conv, sign_expand_kernel);
	result = trainWithRandomForest();

	return result;
}