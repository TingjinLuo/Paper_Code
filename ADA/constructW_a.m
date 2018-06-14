function S = constructW_a(fea,options,expa)

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
    Dist = EuDist2(fea(classIdx,:),[],0);
    D = log(expa)*options.t*expa.^(-options.t*Dist);
    G(classIdx,classIdx) = D;
end

S = sparse(max(G,G'));

% S = max(S,S');