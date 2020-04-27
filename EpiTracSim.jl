# EpiTracSim: simulates the spread of an infection in a
# population of individuals, and evaluates the performance
# of a proposed contact-tracing method. For details of the
# method, see manuscript by Vishwesha Guttal, Sandeep Krishna,
# Rahul Siddharthan (in preparation).

# This code (c) Rahul Siddharthan, 2020.
# Licence: BSD 2-clause.
# Code and demo jupyter notebook available at
# https://github.com/rsidd120/EpiTracSim

module EpiTracSim
using Random

export make_network, read_network_from_file, save_network_to_file,
    get_Ncontact_from_links, make_structures, sweep

# N is the number of nodes. The n'th node is numbered n. Initialize
# all coordination numbers to zero
function create_coordination_numbers(N::Integer)
    a = fill(convert(UInt32,0),N)
    return a
end

struct Node
    node1::UInt32
    node2::UInt32
end

mutable struct Link
    node1::UInt32
    node2::UInt32
    weight::Float32
end

function sample_weight(weights::Array{Float64,1},N)
    n = 1
    w = weights[1]
    r = rand()
    while r > w && n < N
        n += 1
        w += weights[n]
    end
    return n
end

# initially, generate family links. 
# Families of size 1-N, with probability distribution given as array of length N. 
# Links have weight 1, so families are always sampled at every step, and are likely
# to propagate infection to one another.
function generate_links_family!(coordnums::Array{UInt32,1}, linkdict::Dict{Pair,Float32}, weights::Array{Float64,1})
    weights = weights/sum(weights)
    nfam = length(weights)
    N = length(coordnums)
    n = 1
    while n < N
        if n > N-nfam
            famsize = nfam
        else
            famsize = sample_weight(weights,nfam)
        end
        for m = 1:famsize-1
            for p = (m+1):famsize
                linkdict[Pair(m,p)] = convert(Float32,1.0)
                coordnums[m] += 1
                coordnums[p] += 1
            end
        end
        n += famsize
    end
end

# Next, generate links for frequent interactors. 
# Take fraction (eg 1%?) of total, put large number of links (N=10000?) randomly to full population
# Put weight = M/N (M = 100?) so that on average 100 of these links are sampled at each step
function generate_links_bignodes!(coordnums::Array{UInt32,1}, linkdict::Dict{Pair,Float32}, 
        p_interaction::Float64, Ninteractors::Int64, Minteractions::Int64)
    N = length(coordnums)
    interactors = randsubseq(collect(1:N),p_interaction) 
    p = Minteractions/Ninteractors
    for i in interactors
        interactees = randsubseq(collect(1:N),Ninteractors/N) # approximate
        for j in interactees
            if i != j
                if i<j
                    pair1 = Pair(i,j)
                else
                    pair1 = Pair(j,i)
                end
                if haskey(linkdict,pair1)
                    linkdict[pair1] += p
                else
                    linkdict[pair1] = p
                end
                coordnums[i] += 1
                coordnums[j] += 1
            end
        end
    end
end

# now, Barabasi-Albert for remaining interactions
# Not adding any new nodes: re-linking existing nodes 
# for each node, k times,
# 1. subsample n<<N (use n = N*p_sample)
# 2. get probs = coord nums / tot coord nums
# 3. link node to those nodes with that prob
function generate_links_BA!(coordnums::Array{UInt32,1}, linkdict::Dict{Pair, Float32},
    p_sample::Float64, kiter::Int64, weight::Float64)
    N = length(coordnums)
    for k = 1:kiter
        for node in 1:N
            interactors = randsubseq(vcat(collect(1:node-1),collect(node+1:N)),p_sample)
            probs = [coordnums[i] for i in interactors]
            probs /= sum(probs)
            for i in 1:length(interactors)
                if rand() < probs[i]
                    if node < interactors[i]
                        pair1 = Pair(node,interactors[i])
                    else
                        pair1 = Pair(interactors[i],node)
                    end
                    if haskey(linkdict,pair1)
                        linkdict[pair1] += weight
                    else
                        linkdict[pair1] = weight
                    end
                    coordnums[node] += 1
                    coordnums[interactors[i]] += 1
                end
            end
        end
    end
end


function listfromdict(linkdict::Dict{Pair,Float32})
    links::Array{Link,1} = []
    for k in keys(linkdict)
        push!(links,Link(k[1],k[2],linkdict[k]))
    end
    return links
end

function make_network(NNodes::Int64, pop_dist::Array{Float64,1},
    bignode_frac::Float64,bignode_totalint::Int64,bignode_dailyint::Int64,
    ba_samplefrac::Float64,ba_niter::Int64,ba_weight::Float64)
    coordnums = create_coordination_numbers(NNodes);
    linkdict = Dict{Pair,Float32}()
    generate_links_family!(coordnums,linkdict,pop_dist)
    generate_links_bignodes!(coordnums,linkdict,bignode_frac,bignode_totalint, bignode_dailyint)
    generate_links_BA!(coordnums,linkdict,ba_samplefrac,ba_niter,ba_weight)
    links = listfromdict(linkdict);
    return links
end

function make_network_uniform(NNodes::Int64, coord_num::Int64, weight::Float64)
    coordnums = create_coordination_numbers(NNodes)
    linkdict = Dict{Pair,Float32}()
    for n1 = 1:NNodes
        for m = 1:coord_num
            n2 = rand(1:NNodes)
            if n1==n2
                continue
            end
            if n1>n2
                n1,n2 = n2,n1
            end
            if haskey(linkdict,Pair(n1,n2))
                continue
            end
            linkdict[Pair(n1,n2)] = weight
        end
    end
    links = listfromdict(linkdict)
    return links
end

function make_network_alltoall(NNodes::Int64, weight::Float64)
    coordnums = create_coordination_numbers(NNodes)
    linkdict = Dict{Pair,Float32}()
    for n = 1:NNodes-1
        for m = n+1:NNodes
            linkdict[Pair(n1,n2)] = weight
        end
    end
    links = listfromdict(linkdict)
    return links
end

function num_to_coord(n::Int64,dim::Int64)
    x1 = div((n-1), dim^2)
    x2 = div((n-1) % dim^2,dim)
    x3 = (n-1) % dim
    return x1, x2, x3
end

function coord_to_num(x1::Int64,x2::Int64,x3::Int64,dim::Int64)
    return x1*dim^2 + x2*dim + x3 + 1
end

function neighbour(x::Int64,dim::Int64)
    xp = x+1
    if xp>= dim
        xp -= dim
    end
    return xp
end
    
function make_cubic_network(NNodes::Int64, weight::Float64)
    # assume NNodes is a cube
    dim = round(Int64,NNodes^(1/3))
    linkdict = Dict{Pair,Float32}()
    for n in 1:NNodes
        x1,x2,x3 = num_to_coord(n,dim)
        xp = neighbour(x1,dim)
        np = coord_to_num(xp,x2,x3,dim)
        if n < np
            linkdict[Pair(n,np)] = weight
        else
            linkdict[Pair(np,n)] = weight
        end
        xp = neighbour(x2,dim)
        np = coord_to_num(x1,xp,x3,dim)
        if n < np
            linkdict[Pair(n,np)] = weight
        else
            linkdict[Pair(np,n)] = weight
        end
        xp = neighbour(x3,dim)
        np = coord_to_num(x1,x2,xp,dim)
        if n < np
            linkdict[Pair(n,np)] = weight
        else
            linkdict[Pair(np,n)] = weight
        end 
    end
    links = listfromdict(linkdict)
    return links
end    
        
        


function read_network_from_file(filename)
    links::Array{Link,1} = []
    for l in readlines(filename)
        ls = split(l,"\t")
        push!(links,Link(parse(Int32,ls[1]),parse(Int32,ls[2]),parse(Float32,ls[3])))
    end
    return links
end

function save_network_to_file(links,filename)
    f = open("filename","w")
    for l in links
        n = convert(Int,l.node1)
        m = convert(Int,l.node2)
        w = convert(Float64,l.weight)
        write(f,"$n\t$m\t$w\n")
    end
    close(f)
end


mutable struct Contact
    meetings::Array{UInt32,1}
    id::UInt32
end

function make_structures(NNodes::Int64, make_probabilities::Bool)
    contactlist = [[Contact([],0)]]
    pop!(contactlist) # to create a zero-length typed array -- better way?
    for n in 1:NNodes
        C = [Contact([],0)]
        pop!(C)
        push!(contactlist,C)
    end
    if make_probabilities
        probabilities = fill(convert(Float32,0.0),NNodes)
    else
        probabilities = fill(convert(Float32,0.0),0)
    end
    probabilities_naive = fill(convert(Float32,0.0),NNodes);

    infected = fill(convert(Int64,0),NNodes);
    return contactlist,probabilities, probabilities_naive, infected
end


function update_contacts(node::UInt32, sender::UInt32, oldp::Float32, newp::Float32, probabilities::Array{Float32,1},
        contactlist::Array{Array{Contact,1},1}, ignorelist::Array{Int32,1}, tolerance::Float64,p_t::Float64)
    ctime::Int64 = 0
    for d in contactlist[node]
        if d.id==sender
            ctime = d.meetings[1] # since new meetings are pushed, this should be minimum
            break
        end
    end
    downstream = [c for c in contactlist[node] if ~(c.id in ignorelist) && c.meetings[end] >= ctime]
    push!(ignorelist,node)
    for d in downstream
        p_old = probabilities[d.id]
        for n = 1:length([x for x in d.meetings if x>=ctime])
            p_new = convert(Float32,1 - (1 - p_old)*(1-newp*p_t)/(1-oldp*p_t)) 
        end
        # It's possible that this goes below zero because p_old was updated previous to this update
        if p_new < tolerance
            p_new = convert(Float32,0.0)
        end       
 
        #FIXME issue with updating something that is already zero
        if p_new==NaN || p_new < 0 || p_new > 1
            print("p not a number or out of range!\n p_new=",p_new,
                "p_old=",p_old,"   newp=",newp,"   oldp=",oldp, "   p_t=",p_t)
            error("p_new error")
        end
    
        
        probabilities[d.id] = p_new
        if ((p_old == 0.0 && p_new > tolerance) || (p_old > 0.0 && abs(p_new-p_old)/p_old > tolerance))
            update_contacts(d.id,node,p_old,p_new,probabilities,contactlist,ignorelist,tolerance,p_t)
        end
    end
end
            
        

function prune_contacts!(contactlist::Array{Array{Contact,1},1},epoch::Int64,tlimit::Int64, 
        probabilities::Array{Float32,1}, tolerance::Float64)
    for m::UInt32 in 1:length(contactlist)
        c = contactlist[m]
        n = 1
        while n <= length(c)
            c[n].meetings = [x for x in c[n].meetings if x>= epoch-tlimit]
            if length(c[n].meetings) == 0
                #print("Pruning ",c[n])
                c1 = pop!(c)
                c1.meetings = [x for x in c1.meetings if x>= epoch-tlimit]
                while length(c1.meetings)==0 && n <= length(c)
                    c1 = pop!(c) 
                    c1.meetings = [x for x in c1.meetings if x>= epoch-tlimit]
                end
                # DON'T update probability and update_contacts
                if n <= length(c)
                    c[n] = c1
                end
            end
            n += 1
        end
    end
end


function cure_infected!(contactlist::Array{Array{Contact,1},1},
        probabilities::Array{Float32,1},probabilities_naive::Array{Float32,1},
        infected::Array{Int64,1}, epoch::Int64,cure_rate::Float64, tolerance::Float64)
    for n in 1:length(infected)
        if infected[n] > 0 && rand() < cure_rate
            infected[n] = -1
            probabilities_naive[n] = 0.0
            if length(probabilities) > 0
                p_old = probabilities[n]
                probabilities[n] = 0.0
                # no update since they could have picked it up earlier
                #for c in contactlist[n]
                #    ignorelist = [convert(Int32,0) for i in 1:0]
                #    if p_old > 0.0
                #        update_contacts(c.id,convert(UInt32,n),p_old,convert(Float32,0.0),probabilities,contactlist,ignorelist,tolerance)
                #    end
                #end
            end
        end
    end
end


function convert_exposed!(infected::Array{Int64,1}, expose_rate::Float64, epoch::Int64)
    for n in 1:length(infected)
        if infected[n] < -2
            infected[n] += 1
        elseif infected[n] == -2 && rand() <= expose_rate
            infected[n] = 1
        end
    end
end


function test_infected!(links::Array{Link,1},contactlist::Array{Array{Contact,1},1},
        probabilities::Array{Float32,1},probabilities_naive::Array{Float32,1}, 
        infected::Array{Int64,1}, tolerance::Float64, test_threshold::Float64, test_fraction::Float64, 
        isolate_factor::Float64, epoch::Int64, p_t::Float64)
    
    positivelist = []
    for n in 1:length(infected)
        if probabilities[n] > test_threshold 
            if rand() < test_fraction 
                pold = probabilities[n]
                if infected[n] == 1
                    infected[n] = 2 # tested positive
                    probabilities[n] = pnew = probabilities_naive[n] = convert(Float32,1.0)
                    push!(positivelist,n)
                # If negative test, person may still be in risky environment; don't reset to zero?
                elseif infected[n] == 0 || infected[n]<= -2
                    pnew = pold
                    probabilities[n] = pnew = probabilities_naive[n] = convert(Float32,0.0)                    
                else
                    pnew = pold
                end
                if ((pold == 0.0 && pnew > tolerance) || (pold > 0.0 && abs(pnew-pold)/pold > tolerance))
                    for c in contactlist[n]
                        ignorelist = [convert(Int32,0) for i in 1:0]
                        update_contacts(c.id,convert(UInt32,n),pold,pnew,probabilities,contactlist,ignorelist,tolerance,p_t)
                    end
                end
            end
        end
    end
    if isolate_factor < 1.0
        for l in links
            if l.node1 in positivelist || l.node2 in positivelist
                l.weight *= isolate_factor
            end
        end
    end
end


function sweep(links::Array{Link,1},contactlist::Array{Array{Contact,1},1},
        probabilities::Array{Float32,1},probabilities_naive::Array{Float32,1},
        infected::Array{Int64,1}, tolerance::Float64, p_t::Float64, epoch::Int64,tlimit::Int64, 
        cure_rate::Float64, expose_rate::Float64, exposed_init::Int64, miss_rate::Float64,
        test_threshold::Float64, test_fraction::Float64, 
        isolate_factor::Float64)
    for l in links
        if rand() < l.weight
            # update infected and probabilities_naive
            if infected[l.node1] == -1 || infected[l.node2] == -1 # one of them is recovered, immune
                continue
            end
            update_this = (rand() < 1.0-miss_rate) # if false, probabilities not updated
            if infected[l.node1] ==0 && infected[l.node2] > 0 
                if rand() < p_t
                    infected[l.node1] = -2-exposed_init
                end
                if test_threshold == 1.0 && update_this # naive oracle
                    pn = probabilities_naive[l.node1]
                    probabilities_naive[l.node1] = 1-(1-pn)*(1-p_t)
                end                              
            elseif infected[l.node1] > 0 && infected[l.node2] == 0
                if rand() < p_t
                    infected[l.node2] = -2-exposed_init
                end
                if test_threshold == 1.0 && update_this # naive oracle
                    pn = probabilities_naive[l.node2]
                    probabilities_naive[l.node2] = 1-(1-pn)*(1-p_t)
                end                              
            end
            if test_threshold < 1.0 && update_this # naive non-oracle, knows only tested cases
                if infected[l.node1] == 2
                    pn = probabilities_naive[l.node2]
                    probabilities_naive[l.node2] = 1-(1-pn)*(1-p_t)
                end            
                if infected[l.node2] == 2
                    pn = probabilities_naive[l.node1]
                    probabilities_naive[l.node1] = 1-(1-pn)*(1-p_t)
                end
            end
            

            # update probabilities, including contacts
            if length(probabilities) > 0 && update_this
                p1 = probabilities[l.node1]
                p2 = probabilities[l.node2]
                p1new = convert(Float32,1-(1-p1)*(1-p2*p_t))
                p2new = convert(Float32, 1-(1-p2)*(1-p1*p_t))
                if p1new <= tolerance
                    p1new = convert(Float32,0.0)
                end
                if p2new <= tolerance
                    p2new = convert(Float32,0.0)
                end


                probabilities[l.node1] = p1new
                probabilities[l.node2] = p2new 
                # update contact list
                
                # check contacts exist, 0 means doesn't exist
                contact12 = 0
                contact21 = 0
                for n2 in 1:length(contactlist[l.node1])
                    c = contactlist[l.node1][n2]
                    if c.id == l.node2
                        contact12 = n2
                        break
                    end
                end
                for n1 in 1:length(contactlist[l.node2])
                    c = contactlist[l.node2][n1]                
                    if c.id == l.node1
                        contact21 = n1
                        break
                    end
                end
                if contact12 == 0
                    push!(contactlist[l.node1],Contact([epoch],l.node2))
                else
                    push!(contactlist[l.node1][contact12].meetings,epoch)
                end
                if contact21 == 0
                    push!(contactlist[l.node2],Contact([epoch],l.node1))
                else
                    push!(contactlist[l.node2][contact21].meetings,epoch)         
                end
            end
        end
    end
    prune_contacts!(contactlist,epoch,tlimit,probabilities,tolerance)
    if cure_rate > 0
        cure_infected!(contactlist,probabilities,probabilities_naive,infected,epoch,cure_rate,tolerance)
    end
    if test_threshold < 1.0 && length([x for x in infected if x==1]) > 20 # otherwise test kills all infections; make this a parameter
        test_infected!(links,contactlist,probabilities,probabilities_naive, 
            infected, tolerance, test_threshold, test_fraction, 
            isolate_factor, epoch,p_t)
    end
    convert_exposed!(infected, expose_rate, epoch)
    return epoch+1  # this is the time counter
end



function get_Ncontact_from_links(NNodes::Int64,links::Array{Link,1}) 
    Mweights = []
    for i = 1:NNodes
        push!(Mweights,[])
    end
    for l in links
        push!(Mweights[l.node1],l.weight)
        push!(Mweights[l.node2],l.weight)
    end
    return sum([sum(x) for x in Mweights])/NNodes
end


end
