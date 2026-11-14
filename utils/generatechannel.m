function channel_params = generatechannel(num_sats, num_aps, num_ues, antennas_per_sat, antennas_per_ap, Sat_Positions, AP_Positions, UE_Positions, power_fmax)
% Generates channel parameters for the simulation environment.

% Outputs:
% channel_params: A struct containing all the generated channel parameters.

%% System Parameters
% Communication parameters
f = 20000;  % Frequency in MHz
fc = f/1e3; % in GHz
lambdaWave = 3e8/(fc*1e9); % Wavelength in m
dH = lambdaWave; % Antenna spacing
[MH, MV] = find_dimensions(antennas_per_sat); % Antenna array dimensions
B = 100; % Bandwidth in MHz
tau = num_ues; % Number of pilots
tauc = 10000; % Length of coherence block

% AP and user heights
hbs = 35; % AP height in m
hut = 1.5; % user height in m
h = 5; % Average building
W = 20; % Average street width
L1 = 161.04 - 7.1*log10(W) + 7.5*log10(h) - (24.37 - 3.7*(h/hbs)^2)*log10(hbs) ...
  - (3.2*((log10(hut))^2) - 4.97) - (43.42 - 3.1*log10(hbs))*3;

% Power and noise parameters
GA = 5;           % gain at AP in dBi
GSRX = 26.9;      % gain at satellite in dBi
NFA = 7;          % noise figure at AP
NFS = 1.2;        % noise figure at satellite
Pda = power_fmax; % downlink power at AP
Pds = power_fmax; % downlink power at sat
Pua = Pda;        % uplink power at AP
Pus = Pds;        % uplink power at sat
StepVal = 1e19;
sigma2a = StepVal*(10^((-203.975+10*log10(B*10^6)+NFA)/10))/(10^(2*GA/10)); % Noise variance at APs
sigma2s = StepVal*(10^((-203.975+10*log10(B*10^6)+NFS)/10))/(10^((GA+GSRX)/10)); % Noise variance at satellites

% Satellite parameters
RE = 6371e3; % Earth radius in m
h0 = 400e3; % Satellite altitude in m

% Shadow fading parameters
sigma_shda = 8; % in dB
sigma_shds = 2.7; % in dB
xi = ((15*pi)/180)^2; % 15 degree
sigmav1 = xi;

%% Channel Generation
BETAA = zeros(num_aps, num_ues); % Large-scale fading coefficients between users and APs
BETAS = zeros(num_sats, num_ues); % Large-scale fading coefficients between users and satellites (LoS + NLoS)
BETAS_NLoS = zeros(num_sats, num_ues); % Large-scale fading coefficients between users and satellites (NLoS)
RAll = zeros(antennas_per_sat, antennas_per_sat, num_ues, num_sats); % Spatial correlation matrices between users and satellites
KRician = 6.8; % Rician factor
LoS_Channels = zeros(num_ues, antennas_per_sat, num_sats); % Line of sight channels

% AP-UE fading
for ue = 1:num_ues
  for ap = 1:num_aps
    d = norm(AP_Positions(ap, :) - UE_Positions(ue, :));
    betadB = -L1 - 20 * log10(fc) - (43.42 - 3.1 * log10(hbs)) * (log10(d));
    BETAA(ap, ue) = StepVal * 10^(betadB / 10);
  end
end

% Sat channels
for sat = 1:num_sats
  for ue = 1:num_ues
    [Azimuth, Elevation, ~] = cart2sph(Sat_Positions(sat, 1) - UE_Positions(ue, 1), Sat_Positions(sat, 2) - UE_Positions(ue, 2), Sat_Positions(sat, 3));
    d = sqrt(((RE^2) * (sin(Elevation))^2) + h0^2 + 2 * RE * h0) - RE * sin(Elevation);
    betadB = -32.45 - 20 * log10(fc) - 20 * log10(d) + sigma_shds * randn(1, 1);
    
    phik = (pi/32)*rand(1);
    temp = 10*pi*sin(phik);
    Val = 4*abs(besselj(1,temp)/temp)^2;
    
    BETAS(sat, ue) = StepVal * Val * 10^(betadB / 10);
    BETAS_NLoS(sat, ue) = (BETAS(sat, ue) * 1/(KRician + 1));
    
    kSvec = ([cos(Azimuth) * cos(Elevation); sin(Azimuth) * cos(Elevation); sin(Elevation)]') * (2 * pi / lambdaWave);
    hbarrdk = zeros(antennas_per_sat, 1);
    for n = 1:antennas_per_sat
      um = [0; mod(n-1, MH) * dH; floor((n-1)/MH) * dH];
      hbarrdk(n) = exp(1i * kSvec * um);
    end
    LoS_Channels(ue, :, sat) = sqrt((KRician/(KRician + 1)) * BETAS(sat, ue)) * hbarrdk.';
    
    % Calculate spatial correlation matrix
    Correlate_Azimuth = zeros(MH, MH);
    Correlate_Elevation = zeros(MV, MV);
    for m1 = 1:MV
      for m2 = 1:MV
        Temp1 = 1i*(2*pi*dH/lambdaWave)*(m2 - m1)*cos(Elevation);
        Temp2 = -0.5*((xi*2*pi*dH/lambdaWave)^2)*((m2 - m1)^2)*(sin(Elevation)^2);
        Correlate_Elevation(m1,m2) = exp(Temp1)*exp(Temp2);
      end
    end
    for n1 = 1:MH
      for n2 = 1:MH
        D2 = (2*pi*dH/lambdaWave)*(n2 - n1)*sin(Elevation);
        D3 = (xi*2*pi*dH/lambdaWave)*(n2 - n1)*cos(Elevation);
        sigmatilde = (sin(Elevation))*sigmav1;
        D5 = (D3^2*sigmav1^2) + 1;
        Tempv1 = -((D3^2)*(cos(Azimuth))^2)/(2*D5);
        Tempv2 = 1i*D2*cos(Azimuth)/D5;
        Tempv3 = -0.5*((D2*sigmatilde)^2)/D5;
        Correlate_Azimuth(n1,n2) = exp(Tempv1)*exp(Tempv2)*exp(Tempv3)/sqrt(D5);
      end
    end
    RAll(:, :, ue, sat) = kron(Correlate_Elevation, Correlate_Azimuth) * BETAS_NLoS(sat, ue);
  end
end

% Pack results into a struct
channel_params = struct();
channel_params.BETAA = BETAA;
channel_params.BETAS = BETAS;
channel_params.BETAS_NLoS = BETAS_NLoS;
channel_params.RAll = RAll;
channel_params.LoS_Channels = LoS_Channels;
channel_params.sigma2a = sigma2a;
channel_params.sigma2s = sigma2s;
channel_params.Pua = Pua;
channel_params.Pus = Pus;
channel_params.B = B;
channel_params.tau = tau;
channel_params.tauc = tauc;
channel_params.antennas_per_ap = antennas_per_ap;

end