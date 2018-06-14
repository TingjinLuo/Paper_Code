function S = compute_S(fea,options)

% fea = NormalizeFea(fea);
Label = unique(options.gnd);
nLabel = length(Label);

if isfield(options,'gnd') 
    nSmp = length(options.gnd);
else
    nSmp = size(fea,1);
end

G = zeros(nSmp,nSmp);
for i=1:nLabel
    classIdx = find(options.gnd==Label(i));
    D = EuDist2(fea(classIdx,:),[],0);
    D = options.t*exp(-options.t*D);
    G(classIdx,classIdx) = D;
end

for i=1:size(G,1)
    G(i,i) = 0;
end
S = sparse(max(G,G'));

% S = max(S,S');