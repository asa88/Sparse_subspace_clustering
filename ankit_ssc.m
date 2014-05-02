
X=dlmread('tp_embedding.txt');

X=X(18:27,:)';

N = size(X,2);
coords = [cos(2*pi*(1:N)/N); sin(2*pi*(1:N)/N)]';

r = 0; %Enter the projection dimension e.g. r = d*n, enter r = 0 to not project
Cst = 0; %Enter 1 to use the additional affine constraint sum(c) == 1
OptM = 'Lasso'; %OptM can be {'L1Perfect','L1Noise','Lasso','L1ED'}
lambda = 0.001; %Regularization parameter in 'Lasso' or the noise level for 'L1Noise'
K =0 ; %Number of top coefficients to build the similarity graph, enter K=0 for using the whole coefficients
if Cst == 1
    K = max(d1,d2) + 1; %For affine subspaces, the number of coefficients to pick is dimension + 1 
end


Xp = DataProjection(X,r,'NormalProj');
CMat = SparseCoefRecovery(Xp,Cst,OptM,lambda);
CKSym = BuildAdjacency(CMat,K);

CKSym(find(CKSym<1))=0;

gplot(CKSym, coords)
text(coords(:,1) - 0.1, coords(:,2) + 0.1, num2str((1:N)'), 'FontSize', 14)
keyboard
[CMatC,sc,OutlierIndx,Fail] = OutlierDetection(CMat,s);
if (Fail == 0)
    CKSym = BuildAdjacency(CMatC,K);
    [Grps , SingVals, LapKernel] = SpectralClustering(CKSym,n);
    Missrate = Misclassification(Grps,sc);
    save Lasso_001.mat CMat CKSym Missrate SingVals LapKernel Fail
end
