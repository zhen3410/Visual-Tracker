
#include"stdafx.h"
#include<iostream>
#include<fstream>
#include"Visual Tracker.h"
using namespace std;

vector<bbox> Visual_Tracker::run(int choose_number) {
	cout << "***********Video Tracking************" << endl;
	//cout << "Please Choose Video:" << endl;
	cout << "KCF Algorithm!!!" << endl;
	switch (choose_number)
	{
	case 1:
		KCF KCF_tracker;
		vector<bbox> result = KCF_tracker.run(seq);
		return result;
	}
}