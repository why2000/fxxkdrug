for name =  2010:2016
    ini = csvread('./true'+string(name)+'.csv');
    x = ini(:,1);
    y = ini(:,2);
    z = ini(:,3);
    [X,Y,Z]=griddata(x,y,z,linspace(min(x),max(x))',linspace(min(y),max(y)),'v4');
    figure,mesh(X,Y,Z)
    % figure;pcolor(X,Y,Z);shading interp
end
for name =  2010:2015
    ini = csvread('./forward'+string(name)+'.csv');
    x = ini(:,1);
    y = ini(:,2);
    z = ini(:,3);
    [X,Y,Z]=griddata(x,y,z,linspace(min(x),max(x))',linspace(min(y),max(y)),'v4');
    figure,mesh(X,Y,Z)
    % figure;pcolor(X,Y,Z);shading interp
end
% for name =  2016:-1:1990
%     ini = csvread('./backward'+string(name)+'.csv');
%     x = ini(:,1);
%     y = ini(:,2);
%     z = ini(:,3);
%     [X,Y,Z]=griddata(x,y,z,linspace(min(x),max(x))',linspace(min(y),max(y)),'v4');
%     figure,mesh(X,Y,Z)
%     % figure;pcolor(X,Y,Z);shading interp
% end