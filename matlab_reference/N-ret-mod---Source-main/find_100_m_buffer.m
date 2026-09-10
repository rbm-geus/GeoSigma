function buffer_img = find_100_m_buffer(elem_img,direction)

% Function that finds a 100 m buffer around an element defined by a binary
% image


[FX,FY] = gradient(elem_img);

%keyboard
if strcmp(direction,'positive')
    border_img = (abs(FX)+abs(FY))>0;
    buffer_img = border_img>0 & elem_img==0;
elseif strcmp(direction,'negative')
    border_img = (abs(FX)+abs(FY))>0;
    buffer_img = border_img>0 & elem_img==1;    
end


