
include("MKLwrapper.jl")

using LinearAlgebra
using BenchmarkTools
using CUDA
using Printf
using Random

# CUBLAS.cublasLoggerConfigure(1, 0, 1, C_NULL)
# println(BLAS.get_config())

MPMatrix = Union{Matrix{Float64},Matrix{Float32},Matrix{Float16}}



mutable struct Tilemat
    m::Int64
    n::Int64
    tilesize::Int64
    mt::Int64
    nt::Int64
    mats::Array{MPMatrix,2}
    precmask::Matrix{Int32}
    function Tilemat(m::Int64, n::Int64, tilesize::Int64)
        mt = floor(Int64, (m + tilesize - 1) / tilesize)
        nt = floor(Int64, (n + tilesize - 1) / tilesize)
        mats = Array{MPMatrix,2}(undef, mt, nt)
        precmask = zeros(Int32, mt,nt)
        new(m, n, tilesize, mt, nt, mats, precmask)
    end
end

function fillrand!(tilemat::Tilemat)
    tilemat.mats[1, 1] = rand(Float64, tilemat.tilesize, tilemat.tilesize)
end

"""
Convert a lapack dense FP64 matrix into tile.

When converting a matrix from lapack to tile. The dense matrix must be of type Matrix{float64}.
If you want to lower precision, you can choose `lowerprecision` function or convert 
a tile manually.
"""
function lap2tile!(tilemat::Tilemat, denseA::Matrix{Float64})
    Am = size(denseA, 1)
    An = size(denseA, 2)
    tilesize = tilemat.tilesize
    m = tilemat.m
    n = tilemat.n
    mt = tilemat.mt
    nt = tilemat.nt
    @assert Am == m
    @assert An == n
    for j in range(1, nt)
        for i in range(1, mt)
            mm = tilesize
            nn = tilesize
            if i == mt
                mm = min(mm, m - tilesize * (mt - 1))
            end
            if j == nt
                nn = min(nn, n - tilesize * (nt - 1))
            end
            starti = 1 + (i - 1) * tilesize
            startj = 1 + (j - 1) * tilesize
            tilemat.mats[i, j] = denseA[starti:starti+mm-1, startj:startj+nn-1]
        end
    end
end

"""
convert a Tilemat data to lapack format. 

It will convert all tiles to **double** precision.
"""
function tile2lap!(tilemat::Tilemat, denseA::Matrix{Float64})
    Am = size(denseA, 1)
    An = size(denseA, 2)
    tilesize = tilemat.tilesize
    m = tilemat.m
    n = tilemat.n
    mt = tilemat.mt
    nt = tilemat.nt
    @assert Am == m
    @assert An == n
    for j in range(1, nt)
        for i in range(1, mt)
            mm = tilesize
            nn = tilesize
            if i == mt

                mm = min(mm, m - tilesize * (mt - 1))
            end
            if j == nt
                nn = min(nn, n - tilesize * (nt - 1))
            end
            starti = 1 + (i - 1) * tilesize
            startj = 1 + (j - 1) * tilesize
            denseA[starti:starti+mm-1, startj:startj+nn-1] = tilemat.mats[i, j]
        end
    end
end

"""
lower the precision of a tilemat.
"""
function tilesquaresum(mat::MPMatrix)
    tmp = convert(Matrix{Float64}, mat)
    return sum(tmp.^2)
end

function lowerprecision(tilemat::Tilemat, appaccuracy::Float64)
    fro = 0.0
    mt = tilemat.mt
    nt = tilemat.nt
    hf_prec = 5e-3
    tnorms = Array{Float64,2}(undef, mt, nt)
    for i in range(1, mt)
        for j in range(1, nt)
            localfro = tilesquaresum(tilemat.mats[i, j])
            tnorms[i, j] = sqrt(localfro)
            # print(eltype(tilemat.mats[i,j]))
            # print(tnorms[i,j],",,", norm(tilemat.mats[i,j],2))
            # @assert isapprox(tnorms[i,j], norm(tilemat.mats[i,j],2))
            fro += localfro
        end
    end
    fro = sqrt(fro)
    # denseA = zeros(Float64, tilemat.m, tilemat.n)
    # tile2lap!(tilemat,denseA)
    # @assert isapprox(fro, norm(denseA,2))
    threshold = fro * appaccuracy / min(mt, nt)
    sp_prec = mkl_lamch(Float32, 'e')
    for i in range(1, mt)
        for j in range(1, nt)
            if tnorms[i, j] < threshold / hf_prec
                tilemat.mats[i, j] = convert(Matrix{Float16}, tilemat.mats[i, j])
                tilemat.precmask[i,j] = 2
            elseif tnorms[i, j] < threshold / sp_prec
                tilemat.mats[i, j] = convert(Matrix{Float32}, tilemat.mats[i, j])
                tilemat.precmask[i,j] = 1
            end
        end
    end
end


function gemmsimple!(m, k, n)
    A = rand(Float64, m, k)
    B = rand(Float64, k, n)
    C = A * B
    return C
end

function cudagemm(hostA::MPMatrix, hostB::MPMatrix, alpha::Float64, hostC::MPMatrix, beta::Float64)
    datatype = typeof(hostC[1, 1])
    mattype = typeof(hostC)
    cuA = CuArray(convert(mattype, hostA))
    cuB = CuArray(convert(mattype, hostB))
    curalpha = convert(datatype, alpha)
    curbeta = convert(datatype, beta)
    cuC = CuArray(hostC)
    if datatype == Float16
        CUBLAS.gemmEx!('N', 'N', curalpha, cuA, cuB, curbeta, cuC)
    else
        CUBLAS.gemm!('N', 'N', curalpha, cuA, cuB, curbeta, cuC)
    end
    copyto!(hostC, cuC)
end


"""
tile-based mixed precision gemm
"""
function xzgemm_tile!(transA::Int64, transB::Int64, tA::Tilemat, tB::Tilemat, alpha::Float64, tC::Tilemat, beta::Float64)
    if transA == CblasNoTrans && tranB == CblasNoTrans
        @assert tA.m == tC.m && tA.n == tB.m && tB.n == tC.n
        mt = tC.mt
        nt = tC.nt
        kt = tA.nt
        for i in range(1, mt)
            for j in range(1, nt)
                mm = size(tC.mats[i, j], 1)
                nn = size(tC.mats[i, j], 2)
                buffer = convert(typeof(tC.mats[i, j]), zeros(mm, nn))
                for k in range(1, kt)
                    cudagemm(tA.mats[i, k], tB.mats[k, j], 1.0, buffer, 1.0)
                end
                dtype = typeof(tC.mats[i, j][1, 1])
                curalpha = convert(dtype, alpha)
                curbeta = convert(dtype, beta)
                tC.mats[i, j] = curalpha * buffer + curbeta * tC.mats[i, j]
            end
        end
    elseif transA == CblasNoTrans && transB == CblasTrans
        # @assert tA.m == tC.m && tA.n == tB.n && tB.m == tC.n
        # mt = tC.mt
        # nt = tC.nt
        # kt = tA.nt
        # for i in range(1, mt)
        #     for j in range(1, nt)
        #         mm = size(tC.mats[i, j], 1)
        #         nn = size(tC.mats[i, j], 2)
        #         buffer = convert(typeof(tC.mats[i, j]), zeros(mm, nn))
        #         for k in range(1, kt)
        #             cudagemm(tA.mats[i, k], tB.mats[k, j], 1.0, buffer, 1.0)
        #         end
        #         dtype = typeof(tC.mats[i, j][1, 1])
        #         curalpha = convert(dtype, alpha)
        #         curbeta = convert(dtype, beta)
        #         tC.mats[i, j] = curalpha * buffer + curbeta * tC.mats[i, j]
        #     end
        # end
    elseif transA == CblasTrans && transB == CblasNoTrans
        @assert tA.n == tC.m && tA.m == tB.m && tB.n == tC.n
    else
        @assert tA.n == tC.m && tA.m == tB.n && tB.m == tC.n
    end
end

function xzgetrf_tile()

end

function checkgemm(C::Matrix{Float64}, Cref::Matrix{Float64}, A::Matrix{Float64}, B::Matrix{Float64}, 
    alpha::Float64, beta::Float64, K::Int64, eps::Float64)
    Rnorm = norm(C - Cref, 2)
    Anorm = norm(A,2)
    Bnorm = norm(B,2)
    Crefnorm = norm(Cref,2)
    return Rnorm / ((abs(alpha) * max(Anorm, Bnorm) + abs(beta) * Crefnorm) * K * eps)
end



function GemmTest(::Type{T}, transA::Int64, transB::Int64, m::Int64,k::Int64,n::Int64) where {T <: Number} 
    denseA = rand(T, m, k) 
    denseB = rand(T, k, n) 
    denseC = rand(T, m, n) 
    alpha = rand(T)
    beta =  rand(T)
    rC = alpha * denseA * denseB + beta * denseC
    Crefnorm = norm(rC,2)
    mklgemm(T,transA, transB, m, n, k, alpha, beta, denseA, denseB, denseC)
    Rnorm = norm(rC - denseC, 2)
    Anorm = norm(denseA, 2)
    Bnorm = norm(denseB, 2)
    var1 =  mkl_lamch(T,'e')
    res = checkgemm(Rnorm, Anorm, Bnorm,Crefnorm, alpha,beta,k,var1)
    println("GEMM correctness ", res)
    res
end


function matgen_mask(tilemat::Tilemat, precmask::Matrix{Int32}, appaccuracy::Float64)
    sp_prec = mkl_lamch(Float32,'e')
    hf_prec = 5e-3
    nt = min(tilemat.mt, tilemat.nt)
    estimatednorm = tilemat.m * 23 * (1 - convert(Float64,sum(precmask)) / length(precmask))
    threshold = estimatednorm * appaccuracy / nt 
    for i in 1:tilemat.mt
        for j in 1:tilemat.nt
            mm = tilemat.tilesize
            nn = tilemat.tilesize 
            if i == tilemat.mt
                mm = tilemat.m - tilemat.tilesize * (tilemat.mt-1)
            end
            if j == tilemat.nt
                nn = tilemat.n - tilemat.tilesize * (tilemat.nt-1)
            end
            if precmask[i,j] == 0
                tilemat.mats[i,j] = rand(Float64, mm, nn) .* 23.
            elseif precmask[i,j] == 1
                tilemat.mats[i,j] = rand(Float64, mm, nn)
                nrm = norm(tilemat.mats[i,j],2)
                while nrm > threshold / sp_prec
                    tilemat.mats[i,j] = tilemat.mats[i,j] .* 0.1
                    nrm = norm(tilemat.mats[i,j],2)
                end
            elseif precmask[i,j] == 2
                tilemat.mats[i,j] = rand(Float64, mm, nn)
                nrm = norm(tilemat.mats[i,j],2)
                while nrm > threshold / hf_prec
                    tilemat.mats[i,j] = tilemat.mats[i,j] .* 0.1
                    nrm = norm(tilemat.mats[i,j],2)
                end
            end
        end
    end
    # denseA = Matrix{Float64}(undef, tilemat.m, tilemat.n)
    # tile2lap!(tilemat,denseA)
    # println("estimated norm ", estimatednorm, " actually norm ", norm(denseA,2))
    # lowerprecision(tilemat, appaccuracy)
    # display(tilemat.precmask)
    # display(denseA)
end

function genprecmask(m::Int64, n::Int64, tilesize::Int64, sp_percent::Float64, hf_percent::Float64)
    mt = floor(Int32, (m + tilesize - 1) / tilesize)
    nt = floor(Int32, (n + tilesize - 1) / tilesize)
    precarray = zeros(Int32, mt * nt)
    elems_dp = floor(Int64, mt * nt - floor(Int64, mt * nt * sp_percent) - floor(Int64, mt * nt * hf_percent))
    elems_sp = floor(Int64, mt * nt * sp_percent) 
    elems_hf = floor(Int64, mt * nt * hf_percent)
    for i in 1:mt*nt
        if i > elems_dp && i <= elems_sp + elems_dp
            precarray[i] = 1
        elseif i > elems_sp + elems_dp
            precarray[i] = 2
        end
    end
    total_idx = shuffle(precarray)
    precmask = reshape(total_idx, (mt,nt))
    return precmask
end

## write a function
function testxzgemm_tile(m::Int64, k::Int64, n::Int64, tilesize::Int64, appaccuracy::Float64, sp_percent::Float64, hf_percent::Float64)
    @assert sp_percent + hf_percent <= 1.0
    @assert appaccuracy > 0 && appaccuracy < 1.
    tileC = Tilemat(m, n, tilesize) 
    precmask = genprecmask(m,n,tilesize,sp_percent,hf_percent)
    matgen_mask(tileC, precmask, appaccuracy) # tilemat creation is done
    denseC = zeros(Float64, m, n)
    tile2lap!(tileC, denseC)
    display(denseC)
    lowerprecision(tileC, appaccuracy)
    display(tileC.precmask)
    denseA = rand(Float64, m, k)
    tileA = Tilemat(m,k,tilesize)
    lap2tile!(tileA, denseA)
    denseB = rand(Float64, k, n)
    tileB = Tilemat(k,n,tilesize)
    lap2tile!(tileB, denseB)
    alpha::Float64 = 0.123
    beta::Float64 = -2.342
    Cref = alpha * denseA * denseB + beta * denseC
    xzgemm_tile!(CblasNoTrans, CblasNoTrans, tileA, tileB, alpha, tileC, beta)
    Cmp = rand(Float64, m,n)
    tile2lap!(tileC,Cmp)
    display(Cmp)
    display(Cref)
    res = checkgemm(Cmp, Cref, denseA, denseB, alpha, beta, k, appaccuracy)
    println("res, ",res)
end

testxzgemm_tile(8,8,8,2,1e-10,0.9,0.0)