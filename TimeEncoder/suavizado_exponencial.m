x = originalData{59,4}(:,1);
y = originalData{59,4}(:,2);


x_s = x;
y_s = y;

alpha = 0.3;
for i=1:length(x)
    if i==1
        y_s(i) = y(i);
    else
        y_s(i) = alpha*y(i)+(1-alpha)*y_s(i-1);
    end
end

plot(x,y)
hold on 
plot(x_s,y_s)