function landsdel = region_to_landsdel(region)

if strcmp(region,'Jylland')
    landsdel = 1;
end


if strcmp(region,'Fyn')
    landsdel = 2;
end

if strcmp(region,'Fyn-MST')
    landsdel = 22;
end

if strcmp(region,'Sjælland')
    landsdel = 3;
end


if strcmp(region,'AnholtLæsø')
    landsdel = 4;
end
