clc;
clear;

N_IN = 5;
N_HID = 8;
N_OUT = 2;
N_PESOS = N_IN * N_HID + N_HID + N_HID * N_OUT + N_OUT;

function y = sigmoide(x)
    y = 1 ./ (1 + exp(-x));
endfunction

function [W1, b1, W2, b2] = desempacotar(crom)
    N_IN = 5;
    N_HID = 8;
    N_OUT = 2;
    i = 1;
    W1 = matrix(crom(i:i + N_HID * N_IN - 1), N_HID, N_IN);
    i = i + N_HID * N_IN;
    b1 = matrix(crom(i:i + N_HID - 1), N_HID, 1);
    i = i + N_HID;
    W2 = matrix(crom(i:i + N_OUT * N_HID - 1), N_OUT, N_HID);
    i = i + N_OUT * N_HID;
    b2 = matrix(crom(i:i + N_OUT - 1), N_OUT, 1);
endfunction

function x = entradas_rede(c)
    x = [c(1) / 40; c(2) / 40; c(3) / 5; c(4) / 40; c(5) / 5];
endfunction

function [ang, vel] = mlp(crom, x)
    [W1, b1, W2, b2] = desempacotar(crom);
    h = tanh(W1 * x + b1);
    y = sigmoide(W2 * h + b2);
    ang = 10 + 70 * y(1);
    vel = 10 + 25 * y(2);
endfunction

function c = gerar_cenario()
    px = 20 + 20 * rand();
    o1x = 4 + (px * 0.45 - 4) * rand();
    o2x = px * 0.55 + (px * 0.9 - px * 0.55) * rand();
    o1h = 1.5 + 3.5 * rand();
    o2h = 1.5 + 3.5 * rand();
    c = [px, o1x, o1h, o2x, o2h];
endfunction

function col = colide_circulo_ret(x, y, r, rx, rw, rh)
    cx = min(max(x, rx), rx + rw);
    cy = min(max(y, 0), rh);
    col = ((x - cx)^2 + (y - cy)^2) <= r^2;
endfunction

function [acerto, colidiu, dmin, traj_x, traj_y] = simular(ang, vel, c)
    g = 9.8;
    y0 = 1.0;
    rb = 0.3;
    rp = 1.0;
    dt = 0.02;
    nmax = 400;

    acerto = %f;
    colidiu = %f;
    dmin = %inf;

    traj_x = zeros(1, nmax);
    traj_y = zeros(1, nmax);
    theta = ang * %pi / 180;
    fim = nmax;

    for i = 1:nmax
        t = (i - 1) * dt;
        x = vel * cos(theta) * t;
        y = y0 + vel * sin(theta) * t - 0.5 * g * t^2;

        traj_x(i) = x;
        traj_y(i) = y;

        d = sqrt((x - c(1))^2 + (y - 1.0)^2);
        if d < dmin then
            dmin = d;
        end

        if d <= (rb + rp) then
            acerto = %t;
            fim = i;
            break;
        end

        if colide_circulo_ret(x, y, rb, c(2) - 0.5, 1.0, c(3)) | colide_circulo_ret(x, y, rb, c(4) - 0.5, 1.0, c(5)) then
            colidiu = %t;
            fim = i;
            break;
        end

        if y <= 0 then
            fim = i;
            break;
        end
    end

    traj_x = traj_x(1:fim);
    traj_y = traj_y(1:fim);
endfunction

function f = fitness_cenario(acerto, colidiu, dmin)
    if acerto then
        f = 1.0;
    else
        f = 0.9 / (1 + dmin);
        if colidiu then
            f = 0.5 * f;
        end
    end
endfunction

function f = fitness_individuo(crom, cenarios)
    n = size(cenarios, 1);
    soma = 0;
    for i = 1:n
        x = entradas_rede(cenarios(i, :));
        [ang, vel] = mlp(crom, x);
        [acerto, colidiu, dmin] = simular(ang, vel, cenarios(i, :));
        soma = soma + fitness_cenario(acerto, colidiu, dmin);
    end
    f = soma / n;
endfunction

function idx = torneio(fits, k)
    n = length(fits);
    cand = zeros(1, k);
    for i = 1:k
        cand(i) = floor(rand() * n) + 1;
    end
    [vmax, imax] = max(fits(cand));
    idx = cand(imax);
endfunction

function [c1, c2] = sbx(p1, p2, eta, pc)
    n = length(p1);
    c1 = p1;
    c2 = p2;

    if rand() > pc then
        return;
    end

    for i = 1:n
        u = rand();
        if u <= 0.5 then
            beta = (2 * u)^(1 / (eta + 1));
        else
            beta = (1 / (2 * (1 - u)))^(1 / (eta + 1));
        end
        c1(i) = 0.5 * ((1 + beta) * p1(i) + (1 - beta) * p2(i));
        c2(i) = 0.5 * ((1 - beta) * p1(i) + (1 + beta) * p2(i));
    end
endfunction

function m = mutacao(v, pm, sigma)
    m = v;
    for i = 1:length(v)
        if rand() < pm then
            m(i) = m(i) + grand(1, 1, "nor", 0, sigma);
        end
    end
endfunction

function [melhor, hist] = treinar(n_pop, n_ger, n_elite, n_treino, semente)
    N_PESOS = 66;

    rand("seed", semente);
    grand("setsd", semente + 1234);

    pop = -2 + 4 * rand(n_pop, N_PESOS);
    hist_best = zeros(1, n_ger);
    hist_media = zeros(1, n_ger);

    for g = 1:n_ger
        cenarios = zeros(n_treino, 5);
        for i = 1:n_treino
            cenarios(i, :) = gerar_cenario();
        end

        fits = zeros(1, n_pop);
        for i = 1:n_pop
            fits(i) = fitness_individuo(pop(i, :), cenarios);
        end

        hist_best(g) = max(fits);
        hist_media(g) = mean(fits);

        if modulo(g, 10) == 0 then
            mprintf("Geracao %d | melhor = %.5f | media = %.5f\n", g, hist_best(g), hist_media(g));
        end

        [fits_ord, ord] = gsort(fits, "g", "i");

        nova = zeros(n_pop, N_PESOS);
        nova(1:n_elite, :) = pop(ord(1:n_elite), :);

        j = n_elite + 1;
        while j <= n_pop
            p1 = pop(torneio(fits, 3), :);
            p2 = pop(torneio(fits, 3), :);
            [f1, f2] = sbx(p1, p2, 2, 0.9);
            f1 = mutacao(f1, 0.08, 0.3);
            f2 = mutacao(f2, 0.08, 0.3);

            nova(j, :) = f1;
            if j + 1 <= n_pop then
                nova(j + 1, :) = f2;
            end
            j = j + 2;
        end

        pop = nova;
    end

    cenarios = zeros(n_treino, 5);
    for i = 1:n_treino
        cenarios(i, :) = gerar_cenario();
    end

    fits = zeros(1, n_pop);
    for i = 1:n_pop
        fits(i) = fitness_individuo(pop(i, :), cenarios);
    end

    [vmax, ibest] = max(fits);
    melhor = pop(ibest, :);
    hist = struct("best", hist_best, "media", hist_media);
endfunction

function [taxa_acerto, taxa_colisao, ex_cenarios, ex_ang, ex_vel, ex_hit, ex_traj_x, ex_traj_y, ex_traj_n] = avaliar(melhor, n_test, semente)
    rand("seed", semente);
    grand("setsd", semente + 5678);

    acertos = 0;
    colisoes = 0;

    ex_n = min(3, n_test);
    ex_cenarios = zeros(ex_n, 5);
    ex_ang = zeros(1, ex_n);
    ex_vel = zeros(1, ex_n);
    ex_hit = zeros(1, ex_n);
    ex_traj_x = zeros(ex_n, 400);
    ex_traj_y = zeros(ex_n, 400);
    ex_traj_n = zeros(1, ex_n);

    for i = 1:n_test
        c = gerar_cenario();
        [ang, vel] = mlp(melhor, entradas_rede(c));
        [acerto, colidiu, dmin, tx, ty] = simular(ang, vel, c);

        if acerto then
            acertos = acertos + 1;
        end
        if colidiu then
            colisoes = colisoes + 1;
        end

        if i <= ex_n then
            ex_cenarios(i, :) = c;
            ex_ang(i) = ang;
            ex_vel(i) = vel;
            ex_hit(i) = acerto;
            ex_traj_n(i) = length(tx);
            ex_traj_x(i, 1:ex_traj_n(i)) = tx;
            ex_traj_y(i, 1:ex_traj_n(i)) = ty;
        end
    end

    taxa_acerto = 100 * acertos / n_test;
    taxa_colisao = 100 * colisoes / n_test;
endfunction

function plotar_fitness(hist)
    scf(1);
    clf();
    g = (1:length(hist.best))';
    dados = [hist.best' hist.media'];
    plot(g, dados);
    legend(["Melhor"; "Media"], "in_upper_left");
    xtitle("Evolucao da fitness", "Geracao", "Fitness");
    xgrid();
endfunction

function plotar_trajetorias(ex_cenarios, ex_ang, ex_vel, ex_hit, ex_traj_x, ex_traj_y, ex_traj_n)
    n = size(ex_cenarios, 1);
    scf(2);
    clf();

    for i = 1:n
        subplot(n, 1, i);

        k = ex_traj_n(i);
        tx = ex_traj_x(i, 1:k);
        ty = ex_traj_y(i, 1:k);
        c = ex_cenarios(i, :);

        plot(tx, ty);
        xgrid();

        xmax = max([45, c(1) + 3]);
        ymax = max([20, c(3) + 3, c(5) + 3]);
        a = gca();
        a.data_bounds = [0, 0; xmax, ymax];

        xrect(c(2) - 0.5, c(3), 1, c(3));
        xrect(c(4) - 0.5, c(5), 1, c(5));
        xarc(c(1) - 1, 2, 2, 2, 0, 360 * 64);

        if ex_hit(i) then
            status = "acertou";
        else
            status = "errou";
        end

        xtitle(msprintf("Exemplo %d | ang=%.1f deg | vel=%.1f m/s | %s", i, ex_ang(i), ex_vel(i), status), "x (m)", "y (m)");
    end
endfunction

// ==================== Execucao principal ====================
n_pop = 100;
n_ger = 150;
n_elite = 4;
n_treino = 30;
semente = 0;

[melhor, hist] = treinar(n_pop, n_ger, n_elite, n_treino, semente);
save("melhor_pesos.sod", "melhor");

[taxa_acerto, taxa_colisao, ex_cenarios, ex_ang, ex_vel, ex_hit, ex_traj_x, ex_traj_y, ex_traj_n] = avaliar(melhor, 300, semente + 999);

mprintf("\nResultado no teste (300 cenarios):\n");
mprintf("Taxa de acerto: %.2f %%\n", taxa_acerto);
mprintf("Taxa de colisao com obstaculos: %.2f %%\n", taxa_colisao);

plotar_fitness(hist);
plotar_trajetorias(ex_cenarios, ex_ang, ex_vel, ex_hit, ex_traj_x, ex_traj_y, ex_traj_n);
