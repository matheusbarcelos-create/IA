// AngryBirds treinado por Algoritmo Genetico 
// o cromossomo e o vetor de pesos da rede; a fitness e o desempenho no jogo
clear; 
clc;
// parametros da fisica
G = 9.8; Y0 = 1.0; R_BIRD = 0.3; R_PIG = 1.0; PIG_Y = 1.0; OBS_W = 1.0;
T = (0:399)*0.02; NT = length(T);

// faixas das saidas da rede
ANG_MIN = 10; ANG_MAX = 80;
V_MIN = 10;   V_MAX = 35;

// rede 5-8-2
N_IN = 5; N_HID = 8; N_OUT = 2;
N_PESOS = N_IN*N_HID + N_HID + N_HID*N_OUT + N_OUT;

// cenario = [px, o1x, o1h, o2x, o2h]
function c = gerar_cenario()
    px  = 20 + 20*rand();
    o1x = 6 + (px*0.45 - 6)*rand();
    o2x = px*0.5 + (px - 4 - px*0.5)*rand();
    o1h = 1.5 + 3.5*rand();
    o2h = 1.5 + 3.5*rand();
    c = [px, o1x, o1h, o2x, o2h];
endfunction

// entradas normalizadas para ficarem na mesma escala
function X = entradas_rede(C)
    X = [C(:,1)/40, C(:,2)/40, C(:,3)/5, C(:,4)/40, C(:,5)/5];
endfunction

function [W1, b1, W2, b2] = desempacotar(w)
    i = 1;
    W1 = matrix(w(i:i+N_IN*N_HID-1), N_IN, N_HID);   i = i + N_IN*N_HID;
    b1 = w(i:i+N_HID-1);                             i = i + N_HID;
    W2 = matrix(w(i:i+N_HID*N_OUT-1), N_HID, N_OUT); i = i + N_HID*N_OUT;
    b2 = w(i:i+N_OUT-1);
endfunction

function [ang, vel] = mlp(w, X)
    [W1, b1, W2, b2] = desempacotar(w);
    n = size(X, 1);
    oculta = tanh(X*W1 + ones(n,1)*b1);
    saida  = 1 ./ (1 + exp(-(oculta*W2 + ones(n,1)*b2)));
    ang = ANG_MIN + saida(:,1)*(ANG_MAX - ANG_MIN);
    vel = V_MIN   + saida(:,2)*(V_MAX - V_MIN);
endfunction

// primeira posicao verdadeira (NT+1 se nunca acontece)
function i = primeiro(mask)
    k = find(mask, 1);
    if isempty(k) then
        i = length(mask) + 1;
    else
        i = k;
    end
endfunction

function [acertou, bateu_obstaculo, d_min, x, y, i_fim] = simular(ang, vel, c)
    a = ang*%pi/180;
    x = vel*cos(a)*T;
    y = Y0 + vel*sin(a)*T - 0.5*G*T.^2;

    i_chao = primeiro(y <= 0);
    i_obs = NT + 1;
    for k = 1:2
        ox = c(2*k); oh = c(2*k+1);
        bateu = (x >= ox-R_BIRD) & (x <= ox+OBS_W+R_BIRD) & (y <= oh+R_BIRD);
        i_obs = min(i_obs, primeiro(bateu));
    end
    dist = sqrt((x - c(1)).^2 + (y - PIG_Y).^2);
    i_porco = primeiro(dist <= R_PIG + R_BIRD);

    // o voo acaba no primeiro impacto (chao ou obstaculo)
    i_fim = min(i_chao, i_obs);
    acertou = (i_porco <= i_fim) & (i_porco <= NT);
    bateu_obstaculo = (i_obs < i_chao) & ~acertou;
    d_min = min(dist(1:min(i_fim, NT)));
endfunction

function f = fitness_cenario(ang, vel, c)
    [acertou, bateu, d_min] = simular(ang, vel, c);
    if acertou then
        f = 1.0;
    else
        f = 0.9/(1 + d_min);              // quanto mais perto do porco, melhor
        if bateu then f = f*0.5; end      // penaliza bater no obstaculo
    end
endfunction

function f = fitness(w, C, X)
    [ang, vel] = mlp(w, X);
    n = size(C, 1);
    s = 0;
    for k = 1:n
        s = s + fitness_cenario(ang(k), vel(k), C(k,:));
    end
    f = s/n;
endfunction

function ind = torneio(pop, fit, k)
    idx = grand(1, k, "uin", 1, size(pop, 1));
    [m, j] = max(fit(idx));
    ind = pop(idx(j), :);
endfunction

function [f1, f2] = sbx(p1, p2, eta, pc)
    if rand() > pc then
        f1 = p1; f2 = p2;
        return;
    end
    n = length(p1);
    u = rand(1, n);
    beta = zeros(1, n);
    ia = find(u <= 0.5);
    ib = find(u > 0.5);
    beta(ia) = (2*u(ia)).^(1/(eta+1));
    beta(ib) = (1 ./ (2*(1 - u(ib)))).^(1/(eta+1));
    f1 = 0.5*((1+beta).*p1 + (1-beta).*p2);
    f2 = 0.5*((1-beta).*p1 + (1+beta).*p2);
endfunction

function ind = mutacao(ind, pm, sigma)
    n = length(ind);
    mask = bool2s(rand(1, n) < pm);
    ind = ind + mask .* grand(1, n, "nor", 0, sigma);
endfunction

function [melhor, hist] = treinar(n_pop, n_ger, n_elite, n_treino, semente)
    rand("seed", semente); grand("setsd", semente);
    pop = -2 + 4*rand(n_pop, N_PESOS);
    hist = zeros(n_ger, 2);
    for g = 1:n_ger
        // cenarios novos a cada geracao, senao a rede decora (overfitting)
        C = zeros(n_treino, 5);
        for k = 1:n_treino
            C(k,:) = gerar_cenario();
        end
        X = entradas_rede(C);

        fit = zeros(n_pop, 1);
        for i = 1:n_pop
            fit(i) = fitness(pop(i,:), C, X);
        end
        [fit, ordem] = gsort(fit, "g", "d");
        pop = pop(ordem, :);
        hist(g,:) = [fit(1), mean(fit)];
        if modulo(g-1, 10) == 0 | g == n_ger then
            mprintf("Geracao %3d | melhor=%.3f | media=%.3f\n", g-1, fit(1), mean(fit));
        end

        // elitismo: os melhores passam direto
        nova = zeros(n_pop, N_PESOS);
        nova(1:n_elite, :) = pop(1:n_elite, :);
        n = n_elite;
        while n < n_pop
            [f1, f2] = sbx(torneio(pop, fit, 3), torneio(pop, fit, 3), 2, 0.9);
            n = n + 1; nova(n,:) = mutacao(f1, 0.08, 0.3);
            if n < n_pop then
                n = n + 1; nova(n,:) = mutacao(f2, 0.08, 0.3);
            end
        end
        pop = nova;
    end
    melhor = pop(1, :);
endfunction

// teste em cenarios que nao apareceram no treino
function [acerto, colisao, C, ang, vel] = avaliar(w, n, semente)
    rand("seed", semente); grand("setsd", semente);
    C = zeros(n, 5);
    for k = 1:n
        C(k,:) = gerar_cenario();
    end
    [ang, vel] = mlp(w, entradas_rede(C));
    ok = 0; col = 0;
    for k = 1:n
        [a, b] = simular(ang(k), vel(k), C(k,:));
        ok = ok + bool2s(a); col = col + bool2s(b);
    end
    acerto = ok/n; colisao = col/n;
endfunction

// programa principal
[melhor, hist] = treinar(100, 150, 4, 30, 0);
[acerto, colisao, Ct, angt, velt] = avaliar(melhor, 300, 123);
mprintf("\nTeste: acertou %.1f%% | bateu em obstaculo %.1f%%\n", 100*acerto, 100*colisao);

// graficos
scf(0); clf();
subplot(1, 4, 1);
plot(0:size(hist,1)-1, hist(:,1), "b", 0:size(hist,1)-1, hist(:,2), "r");
xtitle("Evolucao da fitness", "geracao", "fitness");
legend("melhor", "media", 4);

for j = 1:3
    c = Ct(j,:);
    [ok, bateu, d, x, y, fim] = simular(angt(j), velt(j), c);
    subplot(1, 4, j+1);
    plot(x(1:min(fim,NT)), y(1:min(fim,NT)), "b");
    for k = 1:2
        xfrect(c(2*k), c(2*k+1), OBS_W, c(2*k+1));
        e = gce(); e.background = color("brown");
    end
    xfarc(c(1)-R_PIG, PIG_Y+R_PIG, 2*R_PIG, 2*R_PIG, 0, 360*64);
    e = gce(); e.background = color("green");
    a = gca(); a.data_bounds = [0 0; 45 20]; a.isoview = "on";
    if ok then s = "ACERTOU"; else s = "ERROU"; end
    xtitle(msprintf("%s | %.0f graus, %.0f m/s", s, angt(j), velt(j)));
end

save("melhor_pesos.sod", "melhor");
