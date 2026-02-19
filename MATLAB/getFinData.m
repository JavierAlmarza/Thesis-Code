tickers = { ...
    '^SP500-50','^SP500-25','^SP500-30','^GSPE','^SP500-40', ...
    '^SP500-35','^SP500-20','^SP500-45','^SP500-15','^SP500-60','^SP500-55'};

startDate = '2011-01-01';
endDate   = '2024-01-01';

allData = table();

for i = 1:numel(tickers)
    url = ['https://query1.finance.yahoo.com/v7/finance/download/' tickers{i} ...
        '?period1=' num2str(posixtime(datetime(startDate))) ...
        '&period2=' num2str(posixtime(datetime(endDate))) ...
        '&interval=1d&events=history&includeAdjustedClose=true'];
    
    opts = weboptions('Timeout', 30);
    raw = webread(url, opts);
    T = readtable(char(url));  % Yahoo returns a CSV
    T.Properties.VariableNames{end} = tickers{i}; % rename Adj Close
    allData = outerjoin(allData, T(:,{'Date',tickers{i}}), ...
        'Keys','Date','MergeKeys',true);
end
