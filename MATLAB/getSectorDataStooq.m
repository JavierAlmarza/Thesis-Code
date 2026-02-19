function dataTbl = getSectorDataStooq()
    % Define tickers (Stooq uses slightly different codes than Yahoo)
    % SPX sector ETFs are not directly on Stooq, but we can use S&P 500 (^SPX),
    % or replace with tickers you want from Stooq.
    % Example here uses SPY (ETF) as a proxy + some US sector ETFs.
    tickers = { ...
        'spy.us',  ... % S&P 500 ETF
        'xly.us',  ... % Consumer Discretionary
        'xlp.us',  ... % Consumer Staples
        'xle.us',  ... % Energy
        'xlf.us',  ... % Financials
        'xlv.us',  ... % Health Care
        'xli.us',  ... % Industrials
        'xlk.us',  ... % Information Technology
        'xlb.us',  ... % Materials
        'xlre.us', ... % Real Estate
        'xlu.us'};     % Utilities

    startDate = datetime(2011,1,1);
    endDate   = datetime(2024,1,1);

    dataTbl = table();

    for i = 1:numel(tickers)
        tkr = tickers{i};
        fprintf("Fetching %s...\n", tkr);

        % Build Stooq URL (CSV download)
        url = sprintf('https://stooq.com/q/d/l/?s=%s&i=d', tkr);

        try
            filename = [tkr '.csv'];
            outfilename = websave(filename, url);
            T = readtable(outfilename);

            % Ensure datetime format
            T.Date = datetime(T.Date, 'InputFormat','yyyy-MM-dd');

            % Keep only desired date range
            T = T(T.Date >= startDate & T.Date <= endDate, :);

            % Rename column
            T.Properties.VariableNames{'Close'} = tkr;

            % Merge into master table
            if isempty(dataTbl)
                dataTbl = T(:, {'Date', tkr});
            else
                dataTbl = outerjoin(dataTbl, T(:, {'Date', tkr}), ...
                                    'Keys','Date','MergeKeys',true);
            end

        catch ME
            warning('Failed to fetch %s: %s', tkr, ME.message);
        end

        pause(1); % polite delay
    end

    % Sort rows by Date
    dataTbl = sortrows(dataTbl, 'Date');

    % Forward-fill missing values (optional)
    dataTbl = fillmissing(dataTbl, 'previous');

    % Compute daily returns
    priceMat = dataTbl{:,2:end};
    returns = priceMat(2:end,:) ./ priceMat(1:end-1,:) - 1;

    % Output: returns matrix aligned with dates
    dataTbl.Returns = [nan(1,size(returns,2)); returns];
end
