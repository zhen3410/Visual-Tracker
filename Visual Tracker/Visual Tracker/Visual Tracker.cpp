
#include"stdafx.h"
#include<iostream>
#include<fstream>
#include"Visual Tracker.h"
using namespace std;

void Visual_Tracker::run(int choose_number) {
	cout << "***********Video Tracking************" << endl;
	//cout << "Please Choose Video:" << endl;
	cout << "KCF Algorithm!!!" << endl;
	switch (choose_number)
	{
	case 1:
		KCF KCF_tracker;
		KCF_tracker.run(seq);
	}
}