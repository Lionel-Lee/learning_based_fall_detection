function [filterValue]  = K_filter(x)
    A = 1;
    H = 1;
    P = 0.1;
    Q = 0.05;
    R = 0.6;
    B = 0.1;
    u = 0;
    filterValue(1) = x(1);
    
    for i = 2:length(x)
        PredictValue = A*filterValue(i-1) + B*u;
        P = A^2*P + Q;
        kalmanGain = P*H/(P*H^2+R);
        filterValue(i) = PredictValue + kalmanGain*(x(i) - PredictValue);
        P = (1 - kalmanGain*H)*P;
    end
end