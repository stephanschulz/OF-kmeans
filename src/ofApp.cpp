#include "ofApp.h"

//https://docs.opencv.org/master/d9/dde/samples_2cpp_2kmeans_8cpp-example.html
//https://docs.opencv.org/master/d1/d5c/tutorial_py_kmeans_opencv.html


using namespace ofxCv;
using namespace cv;

int MAX_CLUSTERS = 5;
int selectedFunction;

Mat img;
RNG rng(12345);

Scalar colorTab[] =
{
    Scalar(0, 0, 255),
    Scalar(0,255,0),
    Scalar(255,100,100),
    Scalar(255,0,255),
    Scalar(0,255,255)
};

string functionNames[] = {"makeNewRandom","makeNewRandom","makeNewGridWithCenters"};

void ofApp::setup() {
    vector<cv::Point2f> myCenters;
    for(int i=0; i<MAX_CLUSTERS;i++){
        myCenters.push_back(cv::Point2f(ofRandom(10,ofGetWidth()-10),ofRandom(10,ofGetHeight()-10)));
    }
    selectedFunction = 2;
    makeNewGridWithCenters(myCenters);
//    makeNewRandom();
    
}

void ofApp::update() {
    
}

void ofApp::draw() {
    
    drawMat(img, 0, 0);
    
    string str;
    str += "keys 1 = "+functionNames[0];
    str += ((selectedFunction == 0) ?  "<-" : "");
    str += "\n";
    str += "keys 2 = "+functionNames[1];
    str += ((selectedFunction == 1) ?  "<-" : "");
    str += "\n";
    str += "keys 3 = "+functionNames[2];
    str += ((selectedFunction == 2) ?  "<-" : "");
    
    ofDrawBitmapStringHighlight(str, 10,10);
}

void ofApp::keyReleased(int key){
    
    if(key == '1'){
        selectedFunction = 0;
        makeNewRandom();
    }
    if(key == '2'){
        selectedFunction = 1;
        makeNewGrid();
    }
    
    if(key == '3'){
        selectedFunction = 2;
        vector<cv::Point2f> myCenters;
        for(int i=0; i<MAX_CLUSTERS;i++){
            myCenters.push_back(cv::Point2f(ofRandom(10,ofGetWidth()-10),ofRandom(10,ofGetHeight()-10)));
        }
        makeNewGridWithCenters(myCenters);
    }
}

//place points in random places and find cluters
void ofApp::makeNewRandom(){
    
    img = Mat(ofGetWidth(), ofGetHeight(), CV_8UC3);
    
    int k, clusterCount = rng.uniform(2, MAX_CLUSTERS+1);
    int i, sampleCount = rng.uniform(1, 1001);
    Mat points(sampleCount, 1, CV_32FC2), labels;
    clusterCount = MIN(clusterCount, sampleCount);
    std::vector<Point2f> centers;
    
    /* generate random sample from multigaussian distribution */
    for( k = 0; k < clusterCount; k++ )
    {
        cv::Point center;
        center.x = rng.uniform(0, img.cols);
        center.y = rng.uniform(0, img.rows);
        Mat pointChunk = points.rowRange(k*sampleCount/clusterCount,
                                         k == clusterCount - 1 ? sampleCount :
                                         (k+1)*sampleCount/clusterCount);
        rng.fill(pointChunk, RNG::NORMAL, Scalar(center.x, center.y), Scalar(img.cols*0.05, img.rows*0.05));
    }
    randShuffle(points, 1, &rng);
    double compactness = kmeans(points, clusterCount, labels,
                                TermCriteria( TermCriteria::EPS+TermCriteria::COUNT, 10, 1.0),
                                3, KMEANS_PP_CENTERS, centers);
    img = Scalar::all(0);
    for( i = 0; i < sampleCount; i++ )
    {
        int clusterIdx = labels.at<int>(i);
        cv::Point ipt = points.at<cv::Point2f>(i);
        cv::circle( img, ipt, 2, colorTab[clusterIdx], FILLED, LINE_AA );
    }
    for (i = 0; i < (int)centers.size(); ++i)
    {
        cv::Point2f c = centers[i];
        cv::circle( img, c, 40, colorTab[i], 1, LINE_AA );
    }
    cout << "Compactness: " << compactness << endl;
}

//place points in an evenly spread grid layout and find clusters
//looks a lot like voronoi
void ofApp::makeNewGrid(){
    
    img = Mat(ofGetWidth(), ofGetHeight(), CV_8UC3);
    
    Mat labels;
    std::vector<Point2f> centers;
    
    std::vector<Point2f> points;
    
    for(int y=0; y<img.rows; y+=10){
        for(int x=0; x<img.cols; x+=10){
            
            points.push_back(cv::Point2f(x,y));
            ofLog()<<points.back().x<<" , "<<points.back().y;
        }
    }
    int sampleCount = points.size();
    
    double compactness = kmeans(points, MAX_CLUSTERS, labels,
                                TermCriteria( TermCriteria::EPS+TermCriteria::COUNT, 10, 1.0),
                                3, KMEANS_PP_CENTERS, centers);
    img = Scalar::all(0);
    for(int i = 0; i < sampleCount; i++ )
    {
        int clusterIdx = labels.at<int>(i);
        cv::Point ipt = points[i];
        cv::circle( img, ipt, 2, colorTab[clusterIdx], FILLED, LINE_AA );
    }
    for (int i = 0; i < (int)centers.size(); ++i)
    {
        cv::Point2f c = centers[i];
        cv::circle( img, c, 40, colorTab[i], 1, LINE_AA );
    }
    cout << "Compactness: " << compactness << endl;
    
}

//not working yet
//place points in an evenly spread grid layout 
//pre-define cluster centers and find point belonging to this clusters
void ofApp::makeNewGridWithCenters(vector<cv::Point2f> _myCenters){
    //https://stackoverflow.com/questions/39803033/how-to-set-initial-centers-of-k-means-opencv-c
    
    //https://stackoverflow.com/questions/52556885/how-to-setup-initial-cente-in-opencv-python-using-cv2-kmeans-use-initial-labels
    //provide both the initial centroids you want and the initial labels for every sample
    
    //https://translate.google.com/translate?sl=auto&tl=en&u=https://www.coder.work/article/1230738

    //https://github.com/opencv/opencv/blob/master/modules/core/src/kmeans.cpp
    
    img = Mat(ofGetWidth(), ofGetHeight(), CV_8UC3);
    //make an even grid 
    std::vector<Point2f> points;
    for(int y=0; y<img.rows; y+=10){
        for(int x=0; x<img.cols; x+=10){
            points.push_back(cv::Point2f(x,y));
        }
    }
 
    //pre seed the labels with 0-(MAX_CLUSTERS-1)
    //this does not make sense
    Mat labels_mat =  Mat(1, points.size(), CV_32S);
    for(int i=0; i<points.size(); i++){
            labels_mat.at<int>(i) = ofMap(i,0,points.size()-1,0,MAX_CLUSTERS-1); //index
    }
    
    //it would make sense to pre-seed the cluster centers but it has no effect
    //centers = _myCenters;
    std::vector<Point2f> centers; // = _myCenters;

    
    double compactness = cv::kmeans(points, MAX_CLUSTERS, labels_mat,
                                TermCriteria( TermCriteria::EPS+TermCriteria::COUNT, 10, 1.0),
                                3, KMEANS_USE_INITIAL_LABELS, centers);

    img = Scalar::all(0);
    
    //draw the grid points
    for(int i = 0; i < points.size(); i++ )
    {
        int clusterIdx = labels_mat.at<int>(i); // labels[i];
        cv::Point ipt = points[i]; // points.at<cv::Point2f>(i);
        cv::circle( img, ipt, 2, colorTab[clusterIdx], FILLED, LINE_AA );
    }
    
    //draw the newly found cluster centers
    for (int i = 0; i < (int)centers.size(); ++i)
    {
        cv::Point2f c = centers[i];
        cv::circle( img, c, 40, colorTab[i], 1, LINE_AA );
    }
    
    //draw the original seeded cluster centers
    for (int i = 0; i < (int)_myCenters.size(); ++i)
    {
        cv::circle( img, _myCenters[i], 4, Scalar(255,255,255), 1, LINE_AA );
    }
    
    cout << "Compactness: " << compactness << endl;
    
}
