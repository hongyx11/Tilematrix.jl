using MKL
using MKL_jll


# const variable
LAPACK_ROW_MAJOR::Int64 = 101
LAPACK_COJ_MAJOR::Int64 = 102
CblasNoTrans::Int64 = 111
CblasTrans::Int64 = 112
CblasConjTrans::Int64 = 113
CblasUpper::Int64 = 121
CblasLower::Int64 = 122
CblasNonUnit::Int64 = 131
CblasUnit::Int64 = 132
CblasLeft::Int64 = 141
CblasRight::Int64 = 142

"""
mkl wrapper of cblas_?nrm2 for s,d.
"""
function mkl_lamch(::Type{T},c::Char) where {T <: Real}
    if T == Float32
        val = ccall((:slamch, libmkl_rt), Float32, (Ref{Cchar},), c)
    else
        val = ccall((:dlamch, libmkl_rt), Float64, (Ref{Cchar},), c)
    end
    return val
end

"""
mkl wrapper of cblas_?nrm2 for s,d.
"""
function mkl_norm(x::Vector{T}) where {T <: Real}
    if typeof(x) == Float32
        ccall((:cblas_dnrm2, libmkl_rt), Float64,
            (Int32, Ref{Float64}, Int32), length(x), x, 1)
    else
        ccall((:cblas_snrm2, libmkl_rt), Float32,
        (Int32, Ref{Float32}, Int32), length(x), x, 1)
    end
end

"""
mkl wrapper of cblas_?gemm for s,d.
"""
function mkl_gemm(::Type{T}, transA::Int64, transB::Int64, m::Int64, n::Int64, k::Int64, alpha::T, beta::T,
    A::Matrix{T}, B::Matrix{T}, C::Matrix{T}) where {T <: Number}
    if T == Float32 
        ccall((:cblas_sgemm, libmkl_rt), Cvoid,
            (Int64, Int64, Int64,
                Int64, Int64, Int64,
                Float32, Ref{Float32}, Int64,
                Ref{Float32}, Int64,
                Float32, Ref{Float32}, Int64),
            102, transA, transB, m, n, k, alpha, A, m, B, k, beta, C, m)
    elseif T == Float64
        ccall((:cblas_dgemm, libmkl_rt), Cvoid,
            (Int64, Int64, Int64,
                Int64, Int64, Int64,
                Float64, Ref{Float64}, Int64,
                Ref{Float64}, Int64,
                Float64, Ref{Float64}, Int64),
            102, transA, transB, m, n, k, alpha, A, m, B, k, beta, C, m)
    else    
        println("unimplemented function.")
        exit()
    end
end
