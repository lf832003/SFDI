function [Y,Xf,Af] = op_3(X,~,~)
%OP_3 neural network simulation function.
%
% Generated by Neural Network Toolbox function genFunction, 04-Apr-2019 11:19:56.
% 
% [Y] = op_3(X,~,~) takes these arguments:
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
x1_step1.xoffset = [3.2387e-13;1.156e-14;5.1291e-15];
x1_step1.gain = [2.18660485863677;2.70888921997539;3.51660717739528];
x1_step1.ymin = -1;

% Layer 1
b1 = [3.1113032167487890689;-1.8164008453261060172;0.28065009325446144128;-2.194306292642279832;0.35976972329733314382;-2.4174962510759359091;2.4746197263095384145;9.0221942299538255838];
IW1_1 = [-0.13716940403003272109 1.2399689314558421938 1.4130152892445426716;1.3091544741104397254 0.031950779227246609016 1.2811684253633797148;-0.068298785741583323272 -0.54423885098096347335 -0.35701628949544900182;-5.9397643504140367199 3.9511136471163843531 0.022985717843667183985;0.012309990638794419959 -0.49327316491698713907 1.1221011791154018589;-1.587303753640998405 1.2534951140438344019 -2.3033224897108963702;0.24165180261454485833 1.3098127539643065198 0.65067306963859616875;4.9517160602320684504 4.1438262714664366726 -0.45066440901140158992];

% Layer 2
b2 = [3.0034385422996328074;1.5719505766836556759;0.81605857766549483312;0.95602884164127932554;-0.86159199159567023685;-0.55962561915447051053;-2.1819751598898671219;-1.8323525653525476464];
LW2_1 = [-0.19337706565217144261 0.051073671705602036175 0.55459607052930581794 2.2668536708956090742 -0.044402968891508923233 0.080698110284615043319 0.090071884189286230415 0.463976476481823219;0.012388291248317490761 0.2605571764316232386 -0.18101553899814534776 0.4474817291719941359 -0.29202960157603380376 -2.9789823415814740137 1.4107334481707836105 -0.32095430029428984042;0.61899053005704918196 -0.041320991233863658965 -0.25154938464372444962 -3.3264608631793186611 -0.69801684294824495325 -0.52493210710512494543 -0.097914565849969159572 -5.7639242486862283243;-0.26598516369497737788 -0.053910921135992134756 0.36741267847598235718 1.0782333302342637094 0.79591504118750788788 -0.089861181242838261696 -0.44114600574380746778 -0.37369679842732733688;-2.2873633339031314016 -0.40722106538324909231 -0.60052992883881273212 -6.3432296398881895527 -2.8769066245463088549 -0.23242987411646606755 -1.374952969627344368 2.8647516058093760449;-1.2997890506598286375 0.027122351277321897489 0.16692387413788109529 -0.93210522615733704122 -0.87508207620408817728 -0.00902091556630471407 -0.053484050079949711776 0.42368601084945484159;-0.40588524617181559462 0.12584151460113296439 0.064234141554266738217 -0.34561625998842809837 0.77994878915321896873 -1.2755586196673347477 0.22494643857672150244 -0.0992776733822714752;-0.89562977682420674697 0.81001779426268794015 -0.5641163079841159389 0.1358345063334450431 -0.19392943222520006574 0.57525692244635850958 0.66135089659780377236 -0.6860480243026455538];

% Layer 3
b3 = [-1.0379960495186570935;1.3052352925860428723;0.36364901228865864624;0.15670410214057597931;-0.1225707754648720782;0.19239058817900195342;0.093701346926102005441;2.1139937689235872575];
LW3_2 = [1.0824755888771719903 0.30053691615872180076 -0.067614487989671040458 0.85885527968671060428 -0.47611289500740544556 0.72718638459205109381 -0.38600078778971880222 -0.17478696551385508595;-0.077476867076883851326 0.52532135754593412003 -0.42378025889720316588 -0.72539917775481210782 -0.99479701863943292839 -0.86012383041977402698 0.8945984579903606404 -0.62753064032452743337;-1.4386308939428935627 -0.7048154118677437685 0.54709741745872308982 -0.24553022209363681294 -0.46394028103828549581 0.89493012803744398376 1.4106061093245594318 1.0084092547467027323;-0.55705248710171439974 0.26007866870660406589 -0.29636842512512184822 0.38763471780639408015 -0.67822712659521122713 -0.94533790729164235422 0.41714880347680305395 0.35717854391738701469;1.860845324411718682 -1.5718832161651592472 2.3493359976645447063 -2.3652149182280273187 5.2694376583642208445 2.0019472114623173553 -0.21648721841254664477 1.167868463884823349;0.23396102293533904692 1.2463020080096716136 4.6812816341615191007 -0.35765405139813932767 -2.8490282804354101209 -0.88167892700368333259 -1.5957004633599352061 0.19466783412097046768;0.82841422757463867299 1.8253512452065536742 5.0728408102407982483 0.25930991060272207127 -2.4423684742466664765 1.1938799620967384119 -0.574755191767432283 -0.43718493830382154641;1.1405330909074713475 0.73939947856646248692 0.26676543676361602619 -0.36289897322509162558 -0.23978100907345756276 -0.9347765962298203446 -0.38829245482290097868 0.37005717109827473976];

% Layer 4
b4 = [-1.7930351350416418033;-0.73039799451404319175;-0.80906372090371347916;-0.55145146759703866923;-0.61288761815257231103;-0.81307621624296022578;1.2397037907850350091;-1.1511236223307019788];
LW4_3 = [0.11627407490381577726 0.42729231432024111781 -0.12070220270214773728 0.31657622667138934913 -0.20023552533955568666 -0.41842502308530438171 1.3021361194270979489 -0.99194435565049743353;-0.77252070013078255606 -0.93059892424570167524 -1.0224927426913124062 -0.83116549183244470278 2.1997299369491578602 0.77098432190085108839 1.8454227225538817336 0.42706749625696482919;0.50615985227485882358 -0.60051880158174097968 -0.5838799166699565335 -1.0356166079858843965 0.29792005244918745577 -0.93289011452709713623 1.9073982143743923867 0.36289918572981105793;0.71535861731372929473 -1.6833124802032668477 -0.089088323510201017363 0.01487834310918603048 0.55220082820293314718 1.0723617771394309273 1.6172961004090407044 0.04095215087596058845;2.9535762112217054387 -0.42305954135739837207 0.78572758833723621574 0.04168243723820694302 -3.1331219442477338255 -5.5685567862178384502 -5.0476704609846905214 -1.2851152948303730117;-0.69156458830038580743 0.048850975180222966365 0.55121187469218124733 1.0977500927322598745 -0.8135277167142618282 1.4097872483013020872 -0.43305368830604706876 -0.49440214004133770098;0.11712215486430158085 -1.0949679817714739194 -0.37381107447684636114 0.61288785681448876286 1.1939000313659966501 0.19979227847618763469 0.18253575109763553463 0.43555994623532395815;-1.7893503789837095219 0.0039608609616885023449 -1.1701907884024429496 0.46646828027745479428 -7.1885570834868026679 0.19534045043198011715 -1.4010261589226635781 1.2516039074223455074];

% Layer 5
b5 = [-1.3977397167722773741;0.75471933148510406131;1.033643439955790333;-0.11653087583273612216;0.32893846761880835006;-0.1678011944638601316;-1.1160888396206989359;-1.3818086213773963511];
LW5_4 = [0.49434984095749801014 1.1441600248791794492 0.18276217817457307557 -1.0122500997552308366 -3.7275725771390111341 -0.34551948754069811143 1.0413940206824776613 -5.0474837238398988504;-0.44040921244191549855 -0.67193830292288703188 -0.34205654057828305259 0.43869736986690544001 7.6137008915723294322 0.55583896825442058454 -1.1050911865433392656 -3.7227166034911709502;-1.1584218817532245804 0.19115988734716882202 0.25691903182781122839 0.35072657062781809056 0.76441170773733480015 -0.1727321484258939055 0.88498538256902126165 0.14056770558481870848;-0.069187933995501535445 -0.67626533692039336731 -1.0680976866719675833 0.29665985001317840553 -0.65411502269600774184 -1.2290028931757026598 1.0247841310379954827 1.3727582875563695275;0.70164554425619451994 1.8859006007881635725 1.7374603942794162315 -0.37954469601174994908 -3.8210728317939088861 0.091249867760497732627 0.56439560157396850837 0.93759326311655655228;-1.1797597439952207665 0.46995933438887260714 1.9594867051707389738 1.6286602942534722516 -2.024631593806303087 0.44448432612469940262 1.2135147059245330325 -3.9918631547001082183;-0.45315982500300383551 -1.2798165467981468257 1.6864259334831857018 -0.01864534111351404469 2.7216573813343347688 -0.13943257709222384166 0.25438850278671198968 3.7935189312090757952;-0.83293768904604881254 0.15833898129199919236 2.0056305878318512192 -1.0137349455468376291 0.56021210081787131685 -0.060987204681164419728 -0.66185283209768641655 -0.53771486014503555584];

% Layer 6
b6 = [-1.8188438334226597615;0.44901983763097252922;1.2901296047637429432;-1.0071454495178804311;0.30898420155324002323;-0.59668627176629407405;0.82695023619598517772;2.2796159615226883055];
LW6_5 = [0.7195131006407408103 -0.55172674257354603622 0.38200307525115911877 -1.1582533384232596152 -0.66002824914368252163 0.85192871854824492317 0.65872492495717793748 -0.56304584341320118579;-2.97344666754222553 5.2174445545939178714 -1.7310398891948550748 0.78144500245506520741 -2.1132211082174920946 -2.0970575541814437948 0.51586590484000172108 0.38077307318084596677;-1.5046916213642740345 -0.52034660375277785871 0.019318475845111295064 -0.8268231503002373417 -1.1481422120240329932 -1.3486973459400446451 -0.73533810725068260439 -0.55766546711937092784;0.18282443024630809503 0.74137515257448716177 -1.2010302332723732377 -0.1254777214302972288 -0.55787524813140876301 -2.5338714555877634993 -1.1341994852642054425 0.63364429017986201309;5.9044441053505849482 -0.6357090891071188743 0.082531695427808249299 -0.36013277714601016344 0.67617133119983185985 3.495036751409501985 -5.2221322844297919374 0.71236461436412268355;-1.2080359086844612015 -0.3794711918289688235 1.0191758374432662304 -0.794578704854159934 -0.18770314854760819512 0.82194723180235018667 0.59614072042269006246 -1.3197563326639329961;0.16697140906480928413 -0.35128307684765613939 -0.8206517678736592325 1.0570984766441622327 1.9508071704452230044 0.068894281465555748389 -0.89268759456297519517 0.046334456293916970182;1.7717997406682399752 -3.8799461191643720426 -0.131086972329222895 -0.88737616869297486488 1.968063358680218089 -0.92647829804698556 -2.4242963611643402189 -1.204640766688309883];

% Layer 7
b7 = [-0.53173137059723185605;0.041759084374967246622];
LW7_6 = [-0.72502057618079740475 5.3540354494138320263 2.5222224535650425992 -0.27921768252835210689 -7.4015748798832614597 1.1444036402951649478 -2.3209199498581241095 -3.745821886309447013;-0.021645721533536988607 0.7629298059814620947 0.59621215707538366413 1.9741016910916622251 -0.059275985102843861685 -1.5854760646512962019 0.13052450450846006524 0.80416118039482420432];

% Output 1
y1_step1.ymin = -1;
y1_step1.gain = [8.03212851405623;0.668896321070234];
y1_step1.xoffset = [0.001;0.01];

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
