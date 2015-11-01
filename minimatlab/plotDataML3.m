function [ ] = plotDataML3( x, y )
    figure;
    hold on;
    for ii=1:size(x,1)
        if (y(ii) == 1)
            color = 'r*';
        elseif (y(ii) == 0)
            color = 'b*';
        end
        plot(x(ii,1), x(ii,2), color);
    end
    hold off;
end

