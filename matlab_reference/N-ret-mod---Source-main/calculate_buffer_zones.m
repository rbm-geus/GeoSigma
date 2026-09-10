function bufz_img2 = calculate_buffer_zones(statelem_bin,nx,ny)
[y,x] = find(statelem_bin == 1);

min_x = max(1,min(x)-4);
max_x = min(nx,max(x)+4);

min_y = max(1,min(y)-4);
max_y = min(ny,max(y)+4);

statelem_temp = statelem_bin(min_y:max_y,min_x:max_x);


% Function to calculate 300 m bufferzones around binary image of
% statistical model element

% Positive part
buffer_img100 = find_100_m_buffer(statelem_temp,'positive');
buffer_img200 = find_100_m_buffer(statelem_temp+buffer_img100,'positive');
buffer_img300 = find_100_m_buffer(statelem_temp+buffer_img100+buffer_img200,'positive');

%bufz_img = statelem_bin+buffer_img100*0.75+buffer_img200*0.5+buffer_img300*0.25;

weights = linspace(0,1,8);

%keyboard 
bufz_img = double(statelem_temp);
bufz_img(buffer_img100) = weights(4);
bufz_img(buffer_img200) = weights(3);
bufz_img(buffer_img300) = weights(2);



% Negative part
buffer_img100 = find_100_m_buffer(statelem_temp,'negative');
buffer_img200 = find_100_m_buffer(statelem_temp-buffer_img100,'negative');
buffer_img300 = find_100_m_buffer(statelem_temp-buffer_img100-buffer_img200,'negative');

bufz_img(buffer_img100) = weights(5);
bufz_img(buffer_img200) = weights(6);
bufz_img(buffer_img300) = weights(7);

bufz_img2 = zeros(ny,nx);
bufz_img2(min_y:max_y,min_x:max_x) = bufz_img;

