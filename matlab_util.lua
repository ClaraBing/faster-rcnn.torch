function meshgrid(x, y)
    local n_row, n_col = (#y)[1], (#x)[1]
    local a, b = torch.Tensor(n_row, n_col), torch.Tensor(n_row, n_col)

    -- Each row of a is a copy of x
    for i=1,n_row do
        a[i] = x
    end
    -- Each col of b is a copy of y
    for j=1,n_col do
        b[{{}, {j}}] = y
    end

    return a,b
end

function bsxfun(f, x, y)
-- element-wise binary operation:
-- f: binary operator (on each row)
-- x: a Tensor
-- y: a Tensor w/ same # of rows as x, or a one-dim Tensor which is then repeated to match x's dimension

    -- debug
    print('in bsxfun: \ntype & size of x:')
    print(type(x))
    print(x:size())
    print('type & size of y:')
    print(type(y))
    print(y:size())

    if x:size() == y:size() then
        print('ERROR: bsxfun: there may be an unneccessary use of bsxfun. Please check.')
        return
    end

    local length = x:size()[1]
    if type(y) == 'number' then
        print('Warning: bsxfun: second arg is a number. May use torch.apply directly.')
        x:apply(function(arg) return f(arg, y) end)
        return x
    end

    -- if x:dim() == y:dim() then

    local output = torch.Tensor(length)
    for i=1,length do
        output[i] = f(x[i], y[i])
    end
    
    return output
end
