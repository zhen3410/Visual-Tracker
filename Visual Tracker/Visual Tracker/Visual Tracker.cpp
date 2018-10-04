
#include"stdafx.h"
#include<iostream>
#include<fstream>
#include"Visual Tracker.h"
using namespace std;

vector<bbox> Visual_Tracker::run(int choose_number,bool visualization) {
	cout << "***********Video Tracking************" << endl;
	//cout << "Please Choose Video:" << endl;
	cout << "KCF Algorithm!!!" << endl;
	switch (choose_number)
	{
	case 1:
		
		KCF KCF_tracker;
		int len = seq->length;
		KCF_tracker.init(seq->ground_truth[0]);
		for (int frame = 0; frame < len; frame++) {
			cv::Mat img = cv::imread(seq->img_path[frame]);
			bbox result=KCF_tracker.run(img, frame);

			if (visualization) {
				cv::Rect box;
				box.x = result.x - result.w / 2;
				box.y = result.y - result.h / 2;
				box.width = result.w;
				box.height = result.h;

				cv::putText(img, std::to_string(frame + 1), cv::Point(20, 40), 6, 1, cv::Scalar(0, 255, 255), 2);
				cv::rectangle(img, box, cv::Scalar(0, 255, 255), 2);
				cv::rectangle(img, seq->ground_truth_otb[frame], cv::Scalar(0, 0, 255), 2);
				cv::imshow("Tracking Video", img);
				cv::waitKey(1);
			}

		}
		return KCF_tracker.res;
	}
}