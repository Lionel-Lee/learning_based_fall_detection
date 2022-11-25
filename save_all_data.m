filename = ["sidefall","forwardfall","backfall"];
lowerthreshold = [320, 220, 95];
upperthreshold = [410, 260, 130];
basePath = sprintf('./data/');
imu_fid = fopen([basePath 'allimu.txt'],'w');
for i=1:length(filename)
    condition = filename(i)
    bag = rosbag(strcat(condition,'.bag'));
    imu_raw = select(bag,'Topic','imu/data');
    imus  = readMessages(imu_raw);
    Vel_x = [];
    Vel_y = [];
    Vel_z = [];
    Acc_x = [];
    Acc_y = [];
    Acc_z = [];
    Ori_x = [];
    Ori_y = [];
    Ori_z = [];

    for j = 1:length(imus)
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
        Vel_x(j) = vel_x;
        Vel_y(j) = vel_y;
        Vel_z(j) = vel_z;
        Acc_x(j) = acc_x;
        Acc_y(j) = acc_y;
        Acc_z(j) = acc_z;
        Ori_x(j) = ori_x;
        Ori_y(j) = ori_y;
        Ori_z(j) = ori_z;
        Ori_w(j) = ori_w;
    end
    
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

    for j = 1:length(Filtered_Vel_x)
        if j < lowerthreshold(i)
            label = 0;
        elseif j > upperthreshold(i)
            label = 0;
        else
            label = 1;
        end
        fprintf(imu_fid, '%f %f %f %f %f %f %f %f %f %f %d\n', Filtered_Acc_x(j),Filtered_Acc_y(j),Filtered_Acc_z(j),Filtered_Vel_x(j),Filtered_Vel_y(j),Filtered_Vel_z(j),Filtered_Ori_x(j),Filtered_Ori_y(j),Filtered_Ori_z(j),Filtered_Ori_w(j), label); 

    end
end
fclose(imu_fid)