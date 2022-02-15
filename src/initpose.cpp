#include "ros/ros.h"
#include "std_msgs/String.h"
#include "geometry_msgs/PoseWithCovarianceStamped.h"
#include "math.h"
#include <cmath>
#include <ros/ros.h>
#include <std_srvs/SetBool.h>
#include <std_srvs/Trigger.h>
#include <std_msgs/Bool.h>
#include <std_msgs/Float32.h>
#include <std_msgs/Int8.h>
#include <gazebo_msgs/LinkStates.h>
#include <gazebo_msgs/ModelStates.h>
#include <gazebo_msgs/SetModelState.h>
#include <vector>
#define PI 3.1415926

int count=0;
double wait_t;
double pose_x=0;
double pose_y=0;

std_srvs::Trigger::Request req;             
std_srvs::Trigger::Response resp;           
std_srvs::SetBool::Request req_ler_str;             
std_srvs::SetBool::Request req_ler_end;             
std_srvs::SetBool::Response resp_ler;           
geometry_msgs::PoseWithCovarianceStamped pose_msg;
ros::ServiceClient Send_Wp_Client; 
ros::ServiceClient StartClient;
ros::ServiceClient set_model_state_client;
ros::ServiceServer srv;
ros::Publisher initial_pose_pub; 
ros::Subscriber reset_sub,pose_sub;
ros::Publisher gazebo_pose_pub;

void initial_pose_set(float pose_x,float pose_y,float ori_z,float ori_w)//initial_pose set function   
{
    pose_msg.header.stamp = ros::Time::now();
    pose_msg.header.frame_id = "map";
    pose_msg.pose.pose.position.x = pose_x;
    pose_msg.pose.pose.position.y = pose_y;
    pose_msg.pose.covariance[0] = 0.25;
    pose_msg.pose.covariance[6 * 1 + 1] = 0.25;
    pose_msg.pose.covariance[6 * 5 + 5] = 0.06853891945200942;
    pose_msg.pose.pose.orientation.z = ori_z;
    pose_msg.pose.pose.orientation.w = ori_w;
    initial_pose_pub.publish(pose_msg);
}

void gazebo_pose_set(std::string model_name, geometry_msgs::Pose pose, geometry_msgs::Twist model_twist)
{
    gazebo_msgs::SetModelState setmodelstate;

    // Model state msg
    gazebo_msgs::ModelState modelstate;
    modelstate.model_name = model_name;
    //modelstate.reference_frame = reference_frame;
    modelstate.pose = pose;
    modelstate.twist = model_twist;
    gazebo_pose_pub.publish(modelstate);

}

void set()
{
    geometry_msgs::Pose model_pose;
    model_pose.position.x = 4;
    model_pose.position.y = 0;
    model_pose.position.z = 0;
    model_pose.orientation.x = 0.0;
    model_pose.orientation.y = 0.0;
    model_pose.orientation.z = 0.0;
    model_pose.orientation.w = 0.0;

    geometry_msgs::Twist model_twist;
    model_twist.linear.x = 0.0;
    model_twist.linear.y = 0.0;
    model_twist.linear.z = 0.0;
    model_twist.angular.x = 0.0;
    model_twist.angular.y = 0.0;
    model_twist.angular.z = 0.0;

    initial_pose_set(4,0,0,0.99);
    gazebo_pose_set("mobile_base", model_pose, model_twist);
    count++;

    ROS_INFO("reset!:%d",count);
}

void Pose_Callback(const geometry_msgs::PoseWithCovarianceStamped &p)
{
    pose_x=p.pose.pose.position.x;
    pose_y=p.pose.pose.position.y;
    if (pose_x >= 9.0 || pose_y <= -1.0 || pose_y >= 1.0)
    {
        set();
        Send_Wp_Client.call(req,resp);
    }     
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "pub_initpose");
    ros::NodeHandle nh;
    ros::Rate loop_rate(10);
    initial_pose_pub = nh.advertise<geometry_msgs::PoseWithCovarianceStamped>("initialpose", 10);//initial_pose publish
    gazebo_pose_pub = nh.advertise<gazebo_msgs::ModelState>("/gazebo/set_model_state", 10);//gazebo_pub
    Send_Wp_Client = nh.serviceClient<std_srvs::Trigger>("send_wp_nav");
    ROS_INFO("ready!");
    pose_sub = nh.subscribe("amcl_pose", 10, &Pose_Callback);
    ros::spin();
    return 0;
}
