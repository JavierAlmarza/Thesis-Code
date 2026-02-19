function dataTbl = getSectorData()
    % Sector tickers (same as your Python script)
    sector_tickers = { ...
        '^SP500-50',  ... % Communication Services
        '^SP500-25',  ... % Consumer Discretionary
        '^SP500-30',  ... % Consumer Staples
        '^GSPE',      ... % Energy
        '^SP500-40',  ... % Financials
        '^SP500-35',  ... % Health Care
        '^SP500-20',  ... % Industrials
        '^SP500-45',  ... % Information Technology
        '^SP500-15',  ... % Materials
        '^SP500-60',  ... % Real Estate
        '^SP500-55'};     % Utilities

    % Date range
    startDate = datetime(2011,1,1);
    endDate   = datetime(2024,1,1);

    % Convert to POSIX timestamps
    startPosix = posixtime(startDate);
    endPosix   = posixtime(endDate);

    % Storage
    dataTbl = table();

    for k = 1:numel(sector_tickers)
        ticker = sector_tickers{k};
        fprintf('Fetching %s...\n', ticker);

        % Build Yahoo Finance CSV URL
        url = ['https://query1.finance.yahoo.com/v7/finance/download/' ...
               strrep(ticker, '^', '%5E') ...
               '?period1=' num2str(floor(startPosix)) ...
               '&period2=' num2str(floor(endPosix)) ...
               '&interval=1d&events=history&includeAdjustedClose=true'];

        % Download and read CSV
        filename = [ticker '_data.csv'];
        try
            outfilename = websave(filename, url);
            tbl = readtable(outfilename);

            % Store Date + Close
            if isempty(dataTbl)
                dataTbl = table(tbl.Date, tbl.Close, 'VariableNames', {'Date', ticker});
            else
                dataTbl = outerjoin(dataTbl, ...
                                    table(tbl.Date, tbl.Close, 'VariableNames', {'Date', ticker}), ...
                                    'Keys', 'Date', 'MergeKeys', true);
            end
        catch ME
            warning('Failed to fetch %s: %s', ticker, ME.message);
        end

        pause(5); % avoid rate limit (429 Too Many Requests)
    end

    if isempty(dataTbl)
        warning('No data fetched (all requests blocked by Yahoo).');
        return
    end

    % Sort rows by Date
    dataTbl = sortrows(dataTbl, 'Date');

    % Save for reuse
    save('sectorData.mat', 'dataTbl');
end
