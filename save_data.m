clear all
clc
condition = 'forwardfall5';
bag = rosbag(strcat(condition,'.bag'));
imu_raw = select(bag,'Topic','imu/data');
imus  = readMessages(imu_raw);
basePath = sprintf('./data/');
imu_fid = fopen([basePath strcat('imu_',condition,'.txt')],'w');
Vel_x = [];
Vel_y = [];
Vel_z = [];
Acc_x = [];
Acc_y = [];
Acc_z = [];
Ori_x = [];
Ori_y = [];
Ori_z = [];

for i = 1:length(imus)
    imu = imus{i,1};
    acc_x = imu.LinearAcceleration.X;
    acc_y = imu.LinearAcceleration.Y;
    acc_z = imu.LinearAcceleration.Z; 
    vel_x = imu.AngularVelocity.X;
    vel_y = imu.AngularVelocity.Y;
    vel_z = imu.AngularVelocity.Z;
    ori_x = imu.Orientation.X;
    ori_y = imu.Orientation.Y;
    ori_z = imu.Orientation.Z;
    ori_w = imu.Orientation.W;
    Vel_x(i) = vel_x;
    Vel_y(i) = vel_y;
    Vel_z(i) = vel_z;
    Acc_x(i) = acc_x;
    Acc_y(i) = acc_y;
    Acc_z(i) = acc_z;
    Ori_x(i) = ori_x;
    Ori_y(i) = ori_y;
    Ori_z(i) = ori_z;
    Ori_w(i) = ori_w;
    
    fprintf(imu_fid, '%f %f %f %f %f %f %f %f %f %f \n', acc_x,acc_y,acc_z,vel_x,vel_y,vel_z,ori_x,ori_y,ori_z,ori_w); 
end
fclose(imu_fid);

imu_filter = fopen([basePath strcat('imu_filter_',condition,'.txt')],'w');
Filtered_Vel_x = kalman_filter(Vel_x);
Filtered_Vel_y = kalman_filter(Vel_y);
Filtered_Vel_z = kalman_filter(Vel_z);
Filtered_Acc_x = kalman_filter(Acc_x);
Filtered_Acc_y = kalman_filter(Acc_y);
Filtered_Acc_z = kalman_filter(Acc_z);
Filtered_Ori_x = kalman_filter(Ori_x);
Filtered_Ori_y = kalman_filter(Ori_y);
Filtered_Ori_z = kalman_filter(Ori_z);
Filtered_Ori_w = kalman_filter(Ori_w);

for i = 1:length(Filtered_Vel_x)
    if i < 337
        label = 0;
    elseif i > 395
        label = 0;
    else
        label = 1;
    end
    fprintf(imu_filter, '%f %f %f %f %f %f %f %f %f %f %d\n', Filtered_Acc_x(i),Filtered_Acc_y(i),Filtered_Acc_z(i),Filtered_Vel_x(i),Filtered_Vel_y(i),Filtered_Vel_z(i),Filtered_Ori_x(i),Filtered_Ori_y(i),Filtered_Ori_z(i),Filtered_Ori_w(i), label); 

end

%sidefall 320-410
%forwardfall 220-260
%backfall 95-130

fclose(imu_filter)
subplot(3,3,1)
plot(Vel_x)
hold
plot(Filtered_Vel_x)
xlabel('Time')
ylabel('Velocity')
title('x-axis velocity')
subplot(3,3,2)
plot(Vel_y)
hold
plot(Filtered_Vel_y)
xlabel('Time')
ylabel('Velocity')
title('y-axis velocity')
subplot(3,3,3)
plot(Vel_z)
hold
plot(Filtered_Vel_z)
xlabel('Time')
ylabel('Velocity')
title('z-axis velocity')
subplot(3,3,4)
plot(Acc_x)
hold
plot(Filtered_Acc_x)
xlabel('Time')
ylabel('Acceleration')
title('x-axis acceleration')
subplot(3,3,5)
plot(Acc_y)
hold
plot(Filtered_Acc_y)
xlabel('Time')
ylabel('Acceleration')
title('y-axis acceleration')
subplot(3,3,6)
plot(Acc_z)
hold
plot(Filtered_Acc_z)
xlabel('Time')
ylabel('Acceleration')
title('z-axis acceleration')
subplot(3,3,7)
plot(Ori_x)
hold
plot(Filtered_Ori_x)
xlabel('Time')
ylabel('Orientation')
title('x-axis orientation')
subplot(3,3,8)
plot(Ori_y)
hold
plot(Filtered_Ori_y)
xlabel('Time')
ylabel('Orientation')
title('y-axis orientation')
subplot(3,3,9)
plot(Ori_z)
hold
plot(Filtered_Ori_z)
xlabel('Time')
ylabel('Orientation')
title('z-axis orientation')
