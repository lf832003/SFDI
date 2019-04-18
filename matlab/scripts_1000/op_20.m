function [Y,Xf,Af] = op_20(X,~,~)
%OP_20 neural network simulation function.
%
% Generated by Neural Network Toolbox function genFunction, 17-Apr-2019 12:02:42.
% 
% [Y] = op_20(X,~,~) takes these arguments:
% 
%   X = 1xTS cell, 1 inputs over TS timesteps
%   Each X{1,ts} = 3xQ matrix, input #1 at timestep ts.
% 
% and returns:
%   Y = 1xTS cell of 1 outputs over TS timesteps.
%   Each Y{1,ts} = 2xQ matrix, output #1 at timestep ts.
% 
% where Q is number of samples (or series) and TS is the number of timesteps.

%#ok<*RPMT0>

% ===== NEURAL NETWORK CONSTANTS =====

% Input 1
x1_step1.xoffset = [3.2387e-13;-0.00087186;-0.0038859];
x1_step1.gain = [2.18660485863677;5.574728595732;14.9503759645796];
x1_step1.ymin = -1;

% Layer 1
b1 = [-4.5132404601905991015;-0.6227863970504713409;0.9116090584566384214;0.24260767895098023228;1.5795123518989346856;0.97648935902277278487;-2.6545572622087130199;-5.3547805617531567535];
IW1_1 = [0.49265197885266981404 -4.3495128200600641222 -0.58451380640819683432;1.3842090730024045886 -0.26880763720473310752 -0.091209679311728589224;-1.0931964990361977552 -1.2453857746636740611 0.24387549855003259891;-1.195843278070769955 0.36497984698312291307 -0.32268499001757228184;1.7411394354410536511 -0.90150506426585186315 0.74695804559864342753;-0.47380700802105502056 2.3187519417809911992 -0.84908691364652943978;-3.9187776845747355203 1.2516432501007557221 0.082962255020696901164;-2.7832897458333074603 -2.4304156733459567619 -0.057508361881117353587];

% Layer 2
b2 = [1.7623159838357187468;1.2461642726875798992;0.25153793261987789087;0.086608636645655739938;0.56467344475297698292;0.11779560339732848828;0.75143148476967724925;1.6701879606909182385];
LW2_1 = [-0.75135312923884878256 -0.21309073537692480027 0.7594361198799722823 0.89535078112301103026 0.81303232519013868274 -0.15553896079395898577 0.13269605027617731308 -0.78481209517881478366;-0.38215971681707766594 0.26282202586644731745 -0.1151809263482052037 1.2528202388711764392 -1.0830430077603763195 0.25077832329906035724 0.16097642661555786048 -0.60549127612545650123;-1.8073348625639975573 0.77774307692037036865 0.33114717401943416508 0.30953204589679705894 -0.60900225898138116154 0.49359236413029328183 3.3917583839114850974 1.6819846248153584867;-3.3294031530295797872 0.89699807424665023436 0.62965960255364805676 -0.36381380496556503523 0.16537476927077821021 2.9214090831358734413 -0.98955025403007679241 -3.9229040865364712687;-0.13858443336937736889 -0.98213175894273607724 0.25403489912360283087 0.49439708190140946309 0.39846707129611813603 -0.51562210326449975373 -0.72902895578066717608 -1.1607391227791845623;-0.30351805390719288624 -0.88679138139316004974 -0.054692040983410177002 -0.75562381137164513767 0.024606779010454275097 0.20956999805017517691 0.24320217194386936632 -0.32223771229980952002;1.1507361897647592208 -0.070247732889018046909 0.32535814454662381756 -0.0068045330813614147633 1.09254697006436019 0.075353130476772681168 -0.65960680477416200329 0.0050352006361231117434;1.7967404942103064602 1.5588855076184942039 0.10617689402145814859 -0.25425958486317162821 1.1366901355446372701 -0.77274504093792917292 -1.6231272462845605098 1.0124688730559814331];

% Layer 3
b3 = [1.839243712483558113;1.3906568732905739694;-0.99584556450172589948;0.63831760397027559328;-0.095775989985033296858;-0.82518809959479788585;-1.0484574490193170337;-1.5809202265377548802];
LW3_2 = [-0.82673417194153320242 0.53944175688657780565 -0.80932953729606949 0.069952907159622115807 -0.020804550648281728426 0.83787605678671595033 0.84626397293169441927 -0.77743618447146001227;-0.34488715151406762471 0.75113631872134201117 -0.96208709814113457792 -0.14517585514182113071 0.60692067458822873682 -0.89986426085645654371 0.29498962918863924632 -0.53528359539577163861;0.77727135271656977444 0.55810610342197075617 -0.21078869529698068819 -0.35006872866213328832 0.81509116776841106233 -1.4245884536191324443 -0.65941505512251641363 0.040577361143749574035;-0.841825580392715489 -0.022565562891714064514 1.3822962418327791045 -5.222535184321658086 1.0412519454926596474 -0.55385162721085567927 0.96110683368515392733 2.1950411712772246453;0.19516247789985807781 0.089098259122357567619 1.9801222479502953444 0.67242534975346568427 0.090795531943418703036 -0.36495787727032180658 -0.18862696771949027941 -0.4647094516414722043;-0.36969637943232913635 -0.7399422021636565816 -0.62349831331915728239 -0.055363812296157832726 -0.78069424863206882126 1.2145675201403103305 -0.51161635478538169952 -0.50599354979885491357;-0.32377061980578458744 0.74013666654256282662 4.4446650686738031766 1.6143269444741965568 0.48650323322489230637 1.1076177288359947415 -0.41739342289919112439 -1.6325845437784400893;-0.72319581493042461862 0.51161152875650017791 -0.4971582915843184014 0.24657892257706362504 -0.16868494088894242511 -0.52214174505383148528 0.7969068708808482171 -0.78432965329526516118];

% Layer 4
b4 = [2.0018330801307828715;-1.4548612390073427658;0.59842993085667095521;-0.038905419869850424441;0.41391682989798017633;0.29400767866552834917;1.3118164789439079598;-1.7013290553846105446];
LW4_3 = [-0.12597606283462378651 -0.53332819158435595952 0.91055459777118163966 0.77616801408271052232 -0.43013792823830587775 -0.010955256433605183031 1.3763395198488519444 1.2855702862732705238;0.36602412178321031 0.57608326248902708144 -0.63015622211900224059 0.91472252904506590188 0.39876128165519941771 1.2617198438857026854 0.30563966120821900452 0.34740610778048292007;-0.91635958441032627775 -0.17403063172126304048 0.44790350590423172461 -0.86351339128094306563 -0.82376862095551284249 -0.59102998623817926127 -1.2578656481921284271 0.71647206444226707589;-0.41955152565006109677 0.61101208069826007385 -1.7995270133428833415 -1.5851172577406837938 -0.088593839954719588725 -0.55372855005731924649 1.0643745477269090927 0.73353108871562289206;1.0018168612085982527 0.58219804135509833909 -0.88720608819989188465 -0.96666906679367747834 0.69832515609306566873 -0.27391384499689119814 -0.031559952479603171904 -0.45205464318051269412;-0.36698008595800996057 0.016692147722585936459 -1.8010826095112164591 -0.19477213264473822174 -0.51285339365612148121 0.5688222766222090776 1.0088072675071477757 0.32038204984392476549;0.64970712755234061309 -0.6457752950568180772 0.49610673141851663326 6.0109931842541186597 -1.7745672848720184245 0.69592267207359692627 -2.6671649210409564823 -0.07699609300084583885;0.15264993060962570026 0.081574896249712189333 0.89317849183018771519 -1.7668809506335028114 -1.1635691407794606089 -1.1621161206015926393 -3.8211285568026589488 -0.22849411550606243959];

% Layer 5
b5 = [-2.0223562992261951976;-1.3748843636773349086;-0.40837919603925992851;-0.046535654884292926292;-0.6735471640609653754;-0.83436141263751328889;1.5313548645584462715;-1.7049757425776075337];
LW5_4 = [0.24512699822056432941 1.2094960759126538452 -0.10149362088603783305 0.1406825923690287472 0.33272073796844653382 -0.75036989941028164885 -0.50912264576084076406 -0.55725343616218203646;0.8620887138575034081 0.67987185024218188545 -0.21387242280486090662 -0.39890857426250525553 0.70844503277691173171 0.47328092508109947945 -0.45376852327640709861 -0.79929146445790710818;-0.79644393354340081714 -0.81979047058942244863 1.9388941650538011885 -0.97877908687755299511 0.55415337230463579399 -0.98269355119968515577 4.1962780218713620073 3.375352805554937774;-1.4873435345874281843 -0.57807332699969571355 -0.68684919536779021065 -1.365944751058060902 -0.02080516902258876602 1.305136900999432914 0.79661442548731142832 0.15582281213724855085;-1.0437137438957726499 0.55737598133831167679 -0.42220515914274880931 -0.80929082727706702638 -0.64187071056611189057 0.058979048751323322197 -0.4317572578676326378 1.6474664041914941492;-0.54242591112995641733 -0.003892854047388420724 -0.32773670501576046554 -1.7404196289205311476 0.93084466649187591614 -0.28542027703866867938 2.1582729311594803789 1.4561806873145530794;-0.61413954182224550493 0.77138942434668189563 0.87090861576321576543 1.4049578335870596124 0.67786960211027524892 0.70359769594687493477 1.3935309206660151382 -1.8150104035713872097;0.30228874703223612697 0.22368356196046768058 -1.1721561546912009355 -1.7573925631058331387 0.79425940701580688419 -0.85276206647220853529 2.4497367616232654797 -1.7856481589417494416];

% Layer 6
b6 = [1.7953152649390202722;-1.3035961047929998102;-0.24898746488042231007;-0.83753598466531165023;-0.72819910686932176613;0.71061549437627768988;0.86692649888365913569;1.7634715696225591319];
LW6_5 = [-0.5191478133486561175 -0.52028228474547399962 -0.41666950702964766773 -0.51827582350319489901 0.70906704391572950019 0.43432032157021976948 0.775536998652552656 0.7364883923783649422;0.91334896781383956288 0.82683290032038248007 0.54853401750052122043 0.18957606805978513553 0.61505493868677463265 0.22665589207632333424 0.97782180969644849622 0.22724423078447356095;-0.37301182648766773031 -0.15417375383822679025 0.7025919819825192647 0.43593463377302749118 0.29476670641363433845 -1.2308734085429609006 2.6321558679686822835 3.0765499685854584833;0.84872574573820902533 -0.11096801273266983234 -1.8255043401664605707 -0.71448335203995905207 -0.70225008716211023696 -1.1900703237633643017 -1.5975451430183280621 -0.94184380825212477006;-0.57316878858178177403 -0.025795724603598666619 -0.80360478931227075083 -0.39187889463074127017 1.5619451974177900322 -0.93198776843951336435 -0.28190555773757897118 0.50573813395323796716;0.95868205736881417778 0.0333010772563906432 0.58172209117403150813 -1.96965127727443301 -0.043157536771772191109 1.0818693873435829556 0.47528751836223931404 0.48967917560766927032;1.1114588319207530098 -0.51095230941287173021 5.3129654269667438982 0.93336718605547175365 -0.65208924406026747 3.0836663781413142793 -0.87655531150904986148 1.4680786020030638372;0.30766964046035760738 0.31531819761291762783 0.96158640219016733308 1.0090973626338233249 -0.29676465780271921169 0.17415273814404760855 0.067390222327664825452 -1.092144404359940868];

% Layer 7
b7 = [0.77407597580043518981;0.2718131190298376243];
LW7_6 = [1.0834528184631480041 -0.54342236160406554024 -3.1045687513304143224 1.4346824445704136064 -0.93864716143278625893 -1.3475994235211379291 -4.6751336295654857267 0.13877019825052575031;-0.09518084143490425364 0.32879638967474628108 -0.069641654357949903109 -0.3823603753761653512 -0.63526049937728346073 -2.1785044186360744689 -0.24130042067266591066 0.26585661428536416784];

% Output 1
y1_step1.ymin = -1;
y1_step1.gain = [8.03212851405623;6.68896321070234];
y1_step1.xoffset = [0.001;0.001];

% ===== SIMULATION ========

% Format Input Arguments
isCellX = iscell(X);
if ~isCellX, X = {X}; end;

% Dimensions
TS = size(X,2); % timesteps
if ~isempty(X)
  Q = size(X{1},2); % samples/series
else
  Q = 0;
end

% Allocate Outputs
Y = cell(1,TS);

% Time loop
for ts=1:TS

    % Input 1
    Xp1 = mapminmax_apply(X{1,ts},x1_step1);
    
    % Layer 1
    a1 = tansig_apply(repmat(b1,1,Q) + IW1_1*Xp1);
    
    % Layer 2
    a2 = tansig_apply(repmat(b2,1,Q) + LW2_1*a1);
    
    % Layer 3
    a3 = tansig_apply(repmat(b3,1,Q) + LW3_2*a2);
    
    % Layer 4
    a4 = tansig_apply(repmat(b4,1,Q) + LW4_3*a3);
    
    % Layer 5
    a5 = tansig_apply(repmat(b5,1,Q) + LW5_4*a4);
    
    % Layer 6
    a6 = tansig_apply(repmat(b6,1,Q) + LW6_5*a5);
    
    % Layer 7
    a7 = repmat(b7,1,Q) + LW7_6*a6;
    
    % Output 1
    Y{1,ts} = mapminmax_reverse(a7,y1_step1);
end

% Final Delay States
Xf = cell(1,0);
Af = cell(7,0);

% Format Output Arguments
if ~isCellX, Y = cell2mat(Y); end
end

% ===== MODULE FUNCTIONS ========

% Map Minimum and Maximum Input Processing Function
function y = mapminmax_apply(x,settings)
  y = bsxfun(@minus,x,settings.xoffset);
  y = bsxfun(@times,y,settings.gain);
  y = bsxfun(@plus,y,settings.ymin);
end

% Sigmoid Symmetric Transfer Function
function a = tansig_apply(n,~)
  a = 2 ./ (1 + exp(-2*n)) - 1;
end

% Map Minimum and Maximum Output Reverse-Processing Function
function x = mapminmax_reverse(y,settings)
  x = bsxfun(@minus,y,settings.ymin);
  x = bsxfun(@rdivide,x,settings.gain);
  x = bsxfun(@plus,x,settings.xoffset);
end
