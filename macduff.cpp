/*
original source: ryanfb/macduff
https://github.com/ryanfb/macduff

re-write source: oasi-adamay
*/

#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#define MACBETH_WIDTH   6
#define MACBETH_HEIGHT  4
#define MACBETH_SQUARES MACBETH_WIDTH * MACBETH_HEIGHT

#define MAX_CONTOUR_APPROX  7

#define MAX_RGB_DISTANCE 444

// BabelColor averages in sRGB:
//   http://www.babelcolor.com/main_level/ColorChecker.htm
// (converted to BGR order for comparison)
cv::Scalar colorchecker_srgb[MACBETH_HEIGHT][MACBETH_WIDTH] =
    {
        {
            cv::Scalar(67,81,115),
            cv::Scalar(129,149,196),
            cv::Scalar(157,123,93),
            cv::Scalar(65,108,90),
            cv::Scalar(176,129,130),
            cv::Scalar(171,191,99)
        },
        {
            cv::Scalar(45,123,220),
            cv::Scalar(168,92,72),
            cv::Scalar(98,84,195),
            cv::Scalar(105,59,91),
            cv::Scalar(62,189,160),
            cv::Scalar(41,161,229)
        },
        {
            cv::Scalar(147,62,43),
            cv::Scalar(72,149,71),
            cv::Scalar(56,48,176),
            cv::Scalar(22,200,238),
            cv::Scalar(150,84,188),
            cv::Scalar(166,136,0)
        },
        {
            cv::Scalar(240,245,245),
            cv::Scalar(201,201,200),
            cv::Scalar(161,161,160),
            cv::Scalar(121,121,120),
            cv::Scalar(85,84,83),
            cv::Scalar(50,50,50)
        }
    };

double euclidean_distance(cv::Scalar p_1, cv::Scalar p_2)
{   
    double sum = 0;
    for(int i = 0; i < 3; i++) {
        sum += pow(p_1.val[i]-p_2.val[i],2.);
    }
    return sqrt(sum);
}

double euclidean_distance(cv::Point2f p_1, cv::Point2f p_2)
{
	return euclidean_distance(cv::Scalar(p_1.x, p_1.y, 0), cv::Scalar(p_2.x, p_2.y, 0));
}

double euclidean_distance(cv::Point p_1, cv::Point p_2)
{
    return euclidean_distance(cv::Scalar(p_1.x,p_1.y,0),cv::Scalar(p_2.x,p_2.y,0));
}

double euclidean_distance_lab(cv::Scalar p_1, cv::Scalar p_2)
{
    // convert to Lab for better perceptual distance
    IplImage * convert = cvCreateImage( cvSize(2,1), 8, 3);
    cvSet2D(convert,0,0,p_1);
    cvSet2D(convert,0,1,p_2);
    cvCvtColor(convert,convert,CV_BGR2Lab);
    p_1 = cvGet2D(convert,0,0);
    p_2 = cvGet2D(convert,0,1);
    cvReleaseImage(&convert);
    
    return euclidean_distance(p_1, p_2);
}

cv::Rect contained_rectangle(cv::RotatedRect box)
{
    cv::Rect2f rect(box.center.x - box.size.width/4,
                  box.center.y - box.size.height/4,
                  box.size.width/2,
                  box.size.height/2);

	return cv::Rect(rect);
}

cv::Scalar rect_average(const cv::Rect rect, const cv::Mat& image)
{       
    cv::Scalar average = cv::Scalar(0);
    int count = 0;
    for(int x = rect.x; x < (rect.x+rect.width); x++) {
        for(int y = rect.y; y < (rect.y+rect.height); y++) {
            if((x >= 0) && (y >= 0) && (x < image.cols) && (y < image.rows)) {
                cv::Scalar s(image.at<cv::Vec3b>(y,x));
                average.val[0] += s.val[0];
                average.val[1] += s.val[1];
                average.val[2] += s.val[2];
            
                count++;
            }
        }
    }
    
    for(int i = 0; i < 3; i++){
        average.val[i] /= count;
    }
    
    return average;
}


cv::Scalar contour_average(std::vector<cv::Point>& contour, cv::Mat& image)
{
	cv::Scalar average = 0;
	cv::Mat mask = cv::Mat(image.size(), CV_8UC1, cv::Scalar(0));
	std::vector<std::vector<cv::Point> > contours;
	contours.push_back(contour);
	cv::drawContours(mask, contours, -1, cv::Scalar(255), CV_FILLED);
	average = cv::mean(image, mask);
	return average;
}


void rotate_box(std::vector<cv::Point2f>& box_corners)
{
	assert(box_corners.size() == 4);
	cv::Point2f last = box_corners[3];
    for(int i = 3; i > 0; i--) {
        box_corners[i] = box_corners[i-1];
    }
    box_corners[0] = last;
}

double check_colorchecker(cv::Mat& colorchecker)
{
    double difference = 0;
    
    for(int x = 0; x < MACBETH_WIDTH; x++) {
        for(int y = 0; y < MACBETH_HEIGHT; y++) {
            cv::Scalar known_value = colorchecker_srgb[y][x];
			cv::Scalar test_value = colorchecker.at<cv::Vec3f>(y,x);
            for(int i = 0; i < 3; i++){
                difference += pow(known_value.val[i]-test_value.val[i],2);
            }
        }
    }
    
    return difference;
}

void draw_colorchecker(cv::Mat& colorchecker_values, cv::Mat& colorchecker_points, cv::Mat& image, int size)
{
    for(int x = 0; x < MACBETH_WIDTH; x++) {
        for(int y = 0; y < MACBETH_HEIGHT; y++) {
            cv::Scalar this_color = colorchecker_values.at<cv::Vec3f>(y,x);
            cv::Point this_point = colorchecker_points.at<cv::Vec2f>(y,x);
            
            cv::circle(
                image,
				this_point,
                size,
                colorchecker_srgb[y][x],
                -1
            );
            
            cv::circle(
                image,
				this_point,
                size/2,
                this_color,
                -1
            );
        }
    }
}

struct ColorChecker {
    double error;
    cv::Mat values;
    cv::Mat points;
    double size;
};



bool find_colorchecker(
	const std::vector< std::vector<cv::Point> >& quads,
	const std::vector< cv::RotatedRect >&  boxes,
	const cv::Mat& image,
	const cv::Mat& original_image,
	ColorChecker& found_colorchecker
)
{
    bool passport_box_flipped = false;
    bool rotated_box = false;
    
	std::vector<cv::Point2f> points(boxes.size());
    for(int i = 0; i < boxes.size(); i++)
    {
		cv::RotatedRect box = boxes[i];
		points[i] = cv::Point2f(box.center.x, box.center.y);
    }

	cv::RotatedRect passport_box = cv::minAreaRect(points);
    fprintf(stderr,"Box:\n\tCenter: %f,%f\n\tSize: %f,%f\n\tAngle: %f\n",passport_box.center.x,passport_box.center.y,passport_box.size.width,passport_box.size.height,passport_box.angle);
    if(passport_box.angle < 0.0) {
      passport_box_flipped = true;
    }

#if 0
	std::vector<cv::Point2f> box_corners(4);
	cv::boxPoints(passport_box, box_corners);
#else
	std::vector<cv::Point2f> box_corners(4);
	cv::Mat _box_corners;
	cv::boxPoints(passport_box, _box_corners);
	for (int i = 0; i < 4; i++) box_corners[i] = cv::Point2f(_box_corners.at<float>(i,0), _box_corners.at<float>(i,1));
#endif

    if(euclidean_distance(box_corners[0],box_corners[1]) <
       euclidean_distance(box_corners[1],box_corners[2])) {
        fprintf(stderr,"Box is upright, rotating\n");
        rotate_box(box_corners);
        rotated_box = true && passport_box_flipped;
    }

    double horizontal_spacing = euclidean_distance(box_corners[0],box_corners[1])/(double)(MACBETH_WIDTH-1);
    double vertical_spacing = euclidean_distance(box_corners[1],box_corners[2])/(double)(MACBETH_HEIGHT-1);
    double horizontal_slope = (box_corners[1].y - box_corners[0].y)/(box_corners[1].x - box_corners[0].x);
    double horizontal_mag = sqrt(1+pow(horizontal_slope,2));
    double vertical_slope = (box_corners[3].y - box_corners[0].y)/(box_corners[3].x - box_corners[0].x);
    double vertical_mag = sqrt(1+pow(vertical_slope,2));
    double horizontal_orientation = box_corners[0].x < box_corners[1].x ? -1 : 1;
    double vertical_orientation = box_corners[0].y < box_corners[3].y ? -1 : 1;
        
    fprintf(stderr,"Spacing is %f %f\n",horizontal_spacing,vertical_spacing);
    fprintf(stderr,"Slope is %f %f\n", horizontal_slope,vertical_slope);
    
    int average_size = 0;
    for(int i = 0; i < boxes.size(); i++)
    {
		cv::RotatedRect box = boxes[i];
        
        cv::Rect rect = contained_rectangle(box);
        average_size += MIN(rect.width, rect.height);
    }
    average_size /= (int)boxes.size();
    
    fprintf(stderr,"Average contained rect size is %d\n", average_size);
    
    cv::Mat this_colorchecker(MACBETH_HEIGHT, MACBETH_WIDTH, CV_32FC3);
    cv::Mat this_colorchecker_points( MACBETH_HEIGHT, MACBETH_WIDTH, CV_32FC2 );
    
    // calculate the averages for our oriented colorchecker
    for(int x = 0; x < MACBETH_WIDTH; x++) {
        for(int y = 0; y < MACBETH_HEIGHT; y++) {
            cv::Point2f row_start;
            
//            if ( ((image->origin == IPL_ORIGIN_BL) || !rotated_box) && !((image->origin == IPL_ORIGIN_BL) && rotated_box) )
            if ((!rotated_box))
			{
                row_start.x = (float)(box_corners[0].x + vertical_spacing * y * (1 / vertical_mag));
                row_start.y = (float)(box_corners[0].y + vertical_spacing * y * (vertical_slope / vertical_mag));
            }
            else
            {
                row_start.x = (float)(box_corners[0].x - vertical_spacing * y * (1 / vertical_mag));
                row_start.y = (float)(box_corners[0].y - vertical_spacing * y * (vertical_slope / vertical_mag));
            }
            
            cv::Rect2f rect = cv::Rect(0,0,average_size,average_size);
            
            rect.x = (float)(row_start.x - horizontal_spacing * x * ( 1 / horizontal_mag ) * horizontal_orientation);
            rect.y = (float)(row_start.y - horizontal_spacing * x * ( horizontal_slope / horizontal_mag ) * vertical_orientation);
            
            this_colorchecker_points.at<cv::Vec2f>(y,x) = cv::Vec2f(rect.x,rect.y);
            
            rect.x = rect.x - average_size / 2;
            rect.y = rect.y - average_size / 2;
            
            
            cv::Scalar average_color = rect_average(rect, original_image);
			this_colorchecker.at<cv::Vec3f>(y, x) = cv::Vec3f((float)average_color[0], (float)average_color[1], (float)average_color[2]);

        }
    }
    
    double orient_1_error = check_colorchecker(this_colorchecker);
	cv::flip(this_colorchecker, this_colorchecker,-1);
    double orient_2_error = check_colorchecker(this_colorchecker);
    
    fprintf(stderr,"Orientation 1: %f\n",orient_1_error);
    fprintf(stderr,"Orientation 2: %f\n",orient_2_error);
    
    if(orient_1_error < orient_2_error) {
		cv::flip(this_colorchecker, this_colorchecker, -1);
    }
    else {
		cv::flip(this_colorchecker_points, this_colorchecker_points, -1);
    }
    
    // draw_colorchecker(this_colorchecker,this_colorchecker_points,image,average_size);
   
    found_colorchecker.error = MIN(orient_1_error,orient_2_error);
    found_colorchecker.values = this_colorchecker;
    found_colorchecker.points = this_colorchecker_points;
    found_colorchecker.size = average_size;
    
    return true;
}

bool find_quad(const std::vector<cv::Point>& contour, std::vector<cv::Point>& quad_contour, int min_size) {
	quad_contour.resize(0);

	std::vector<cv::Point> approx;

	double apprrox_ratio = 0.1;

	// approximate contour with accuracy proportional
	// to the contour perimeter
	cv::approxPolyDP(contour, approx, cv::arcLength(contour, true)*apprrox_ratio, true);

	// reject non-quadrangles
	if (approx.size() == 4 && cv::isContourConvex(cv::Mat(approx))) {

		cv::Point pt[4];
		double d1, d2, p = cv::arcLength(approx, true);
		double area = fabs(cv::contourArea(approx));
		double dx, dy;

		for (int i = 0; i < 4; i++)
			pt[i] = approx[i];

		dx = pt[0].x - pt[2].x;
		dy = pt[0].y - pt[2].y;
		d1 = sqrt(dx*dx + dy*dy);

		dx = pt[1].x - pt[3].x;
		dy = pt[1].y - pt[3].y;
		d2 = sqrt(dx*dx + dy*dy);

		// philipg.  Only accept those quadrangles which are more square
		// than rectangular and which are big enough
		double d3, d4;
		dx = pt[0].x - pt[1].x;
		dy = pt[0].y - pt[1].y;
		d3 = sqrt(dx*dx + dy*dy);
		dx = pt[1].x - pt[2].x;
		dy = pt[1].y - pt[2].y;
		d4 = sqrt(dx*dx + dy*dy);
		if (
			(d3*1.1 > d4 && d4*1.1 > d3 && d3*d4 < area*1.5 && area > min_size &&
				d1 >= 0.15 * p && d2 >= 0.15 * p))
		{
			quad_contour = approx;
			return true;
		}
	}

	return false;
}


bool find_macbeth( cv::Mat& src, cv::Mat& dst )
{
	cv::Mat macbeth_original = src;
	cv::Mat macbeth_img = src.clone();

	std::vector < cv::Mat >macbeth_split(3);
	std::vector < cv::Mat >macbeth_split_thresh(3);

	cv::split(macbeth_img, macbeth_split);
	cv::split(macbeth_img, macbeth_split_thresh);

   
    {
        int adaptive_method = CV_ADAPTIVE_THRESH_MEAN_C;
        int threshold_type = CV_THRESH_BINARY_INV;
        int block_size = cvRound(
            MIN(macbeth_img.cols,macbeth_img.rows)*0.02)|1;
        fprintf(stderr,"Using %d as block size\n", block_size);
        
        double offset = 6;
        
        // do an adaptive threshold on each channel
        for(int i = 0; i < 3; i++) {
            cv::adaptiveThreshold(macbeth_split[i], macbeth_split_thresh[i], 255, adaptive_method, threshold_type, block_size, offset);
        }
        
        cv::Mat adaptive(cv::Size(macbeth_img.cols, macbeth_img.rows), CV_8UC1);
        
        // OR the binary threshold results together
        cv::bitwise_or(macbeth_split_thresh[0],macbeth_split_thresh[1],adaptive);
		cv::bitwise_or(macbeth_split_thresh[2],adaptive,adaptive);
        
                
        int element_size = (block_size/10)+2;
        fprintf(stderr,"Using %d as element size\n", element_size);
        
        // do an opening on the threshold image
        cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT ,cv::Size(element_size,element_size));
		cv::morphologyEx(adaptive,adaptive,cv::MORPH_OPEN,element);


		std::vector<std::vector<cv::Point> > initial_quads;
		std::vector<cv::RotatedRect > initial_boxes;

		// find contours in the threshold image
		std::vector<std::vector<cv::Point> > contours;
		std::vector<cv::Vec4i> hierarchy;
		cv::findContours(adaptive,contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_TC89_L1);

		cv::Mat dbg;
		cv::cvtColor(adaptive, dbg, CV_GRAY2BGR);

        int min_size = (macbeth_img.cols*macbeth_img.rows)/
            (MACBETH_SQUARES*100);
        
        if(contours.size()!=0) {
			for (int i = 0; i < contours.size(); i++) {
				cv::Vec4i info = hierarchy[i];
				int childId = info[2];
				int parentId = info[3];
				bool is_hole_contour = childId < 0 && parentId >= 0;

				if (!is_hole_contour) continue;
				double area = fabs(cv::contourArea(cv::Mat(contours[i])));

				// only interested in contours with these restrictions
				if (area < min_size) continue;

				// only interested in quad-like contours
				std::vector<cv::Point> quad_contour;
				find_quad(contours[i], quad_contour, min_size);
				if (quad_contour.empty())continue;

				initial_quads.push_back(quad_contour);

				cv::RotatedRect box = cv::minAreaRect(quad_contour);
				initial_boxes.push_back(box);


				// fprintf(stderr,"Center: %f %f\n", box.center.x, box.center.y);

				cv::Scalar average = contour_average(quad_contour, macbeth_img);
				double min_distance = MAX_RGB_DISTANCE;
				cv::Point closest_color_idx = cv::Point(-1, -1);
				for (int y = 0; y < MACBETH_HEIGHT; y++) {
					for (int x = 0; x < MACBETH_WIDTH; x++) {
						double distance = euclidean_distance_lab(average, colorchecker_srgb[y][x]);
						if (distance < min_distance) {
							closest_color_idx.x = x;
							closest_color_idx.y = y;
							min_distance = distance;
						}
					}
				}

				cv::Scalar closest_color = colorchecker_srgb[closest_color_idx.y][closest_color_idx.x];

			}

			fprintf(stderr, "%d initial quads found", (int)initial_quads.size());
			{
				cv::Scalar color = cv::Scalar(0, 255, 255);	//Y
				cv::drawContours(dbg, initial_quads, -1, color);
			}


			ColorChecker found_colorchecker;

            if(initial_quads.size() > MACBETH_SQUARES) {
                fprintf(stderr," (probably a Passport)\n");

				cv::Mat points( (int)initial_quads.size() , 1, CV_32FC2 );
                cv::Mat clusters((int)initial_quads.size(), 1, CV_32SC1 );

				std::vector< std::vector<std::vector<cv::Point> > > partitioned_quads(2);
				std::vector< std::vector<cv::RotatedRect > > partitioned_boxes(2);

                
                // set up the points sequence for cvKMeans2, using the box centers
                for(int i = 0; i < initial_quads.size(); i++) {
                    cv::RotatedRect box = initial_boxes[i];
					points.at<cv::Vec2f>(i) = cv::Vec2f(box.center.x, box.center.y);
                }
                
                // partition into two clusters: passport and colorchecker
                cv::kmeans( points, 2, clusters, 
                           cv::TermCriteria( CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, 10, 1.0 ) ,
					1,	cv::KMEANS_PP_CENTERS	);
        

                for(int i = 0; i < initial_quads.size(); i++) {
					int cluster_idx = clusters.at<int>(i);
					partitioned_quads[cluster_idx].push_back(initial_quads[i]);
					partitioned_boxes[cluster_idx].push_back(initial_boxes[i]);

                }
                
                ColorChecker partitioned_checkers[2];
                
                // check each of the two partitioned sets for the best colorchecker
                for(int i = 0; i < 2; i++) {
	                find_colorchecker(partitioned_quads[i], partitioned_boxes[i],
                                      macbeth_img, macbeth_original, partitioned_checkers[i]);
                }
                
                // use the colorchecker with the lowest error
                found_colorchecker = partitioned_checkers[0].error < partitioned_checkers[1].error ?
                    partitioned_checkers[0] : partitioned_checkers[1];
                
            }
            else { // just one colorchecker to test
                fprintf(stderr,"\n");
                find_colorchecker(initial_quads, initial_boxes,
                                 macbeth_img, macbeth_original, found_colorchecker);
            }
            
            // render the found colorchecker
            draw_colorchecker(found_colorchecker.values,found_colorchecker.points,macbeth_img,(int)found_colorchecker.size);


			// print out the colorchecker info
            for(int y = 0; y < MACBETH_HEIGHT; y++) {            
                for(int x = 0; x < MACBETH_WIDTH; x++) {
                    cv::Vec3f this_value = found_colorchecker.values.at<cv::Vec3f>(y, x);
					cv::Vec2f this_point = found_colorchecker.points.at<cv::Vec2f>(y, x);
                    
                    printf("%.0f,%.0f,%.0f,%.0f,%.0f\n",
                        this_point.val[0],this_point.val[1],
                        this_value.val[2],this_value.val[1],this_value.val[0]);
                }
            }
            printf("%0.f\n%f\n",found_colorchecker.size,found_colorchecker.error);
        }


		dst = macbeth_img;
        
		return true;
	}


    return false;
}

int main( int argc, char *argv[] )
{
    if( argc < 2 )
    {
        fprintf( stderr, "Usage: %s image_file [output_image]\n", argv[0] );
        return 1;
    }

    const char *img_file = argv[1];
	cv::Mat imgSrc = cv::imread(img_file);
	cv::Mat imgDst;

    find_macbeth(imgSrc, imgDst );


    if( argc == 3) {
        cv::imwrite( argv[2], imgDst);
    }

#ifdef _DEBUG
	{
		cv::imshow(img_file, imgDst);
		cv::waitKey();
	}
#endif

    return 0;
}

