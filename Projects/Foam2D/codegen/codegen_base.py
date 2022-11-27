import casadi as ca
import os


def gen_code(ident, args, diffvar, Obj, opt=3):
    opts = dict(with_header=True)

    ca_value = ca.Function('ca_{}'.format(ident), args, [Obj])
    ca_value.generate('ca_{}'.format(ident), opts)
    print('compiling generated code for {}...'.format(ident))
    cmd = 'gcc -fPIC -shared -O{} ca_{}.c -o libca_{}.so'.format(opt, ident, ident)
    status = os.system(cmd)
    if status != 0:
        raise Exception('Command {} failed'.format(cmd))
    cmd = 'gcc -fPIC -shared -O{} ca_{}.c -o ca_{}.so'.format(opt, ident, ident)
    status = os.system(cmd)
    if status != 0:
        raise Exception('Command {} failed'.format(cmd))

    ca_gradient = ca.Function('ca_{}_gradient'.format(ident), args, [ca.jacobian(Obj, diffvar)])
    ca_gradient.generate('ca_{}_gradient'.format(ident), opts)
    print('compiling generated code for gradient of {}...'.format(ident))
    cmd = 'gcc -fPIC -shared -O{} ca_{}_gradient.c -o libca_{}_gradient.so'.format(opt, ident, ident)
    status = os.system(cmd)
    if status != 0:
        raise Exception('Command {} failed'.format(cmd))
    cmd = 'gcc -fPIC -shared -O{} ca_{}_gradient.c -o ca_{}_gradient.so'.format(opt, ident, ident)
    status = os.system(cmd)
    if status != 0:
        raise Exception('Command {} failed'.format(cmd))

    ca_hessian = ca.Function('ca_{}_hessian'.format(ident), args, [ca.hessian(Obj, diffvar)[0]])
    ca_hessian.generate('ca_{}_hessian'.format(ident), opts)
    print('compiling generated code for hessian of {}...'.format(ident))
    cmd = 'gcc -fPIC -shared -O{} ca_{}_hessian.c -o libca_{}_hessian.so'.format(opt, ident, ident)
    status = os.system(cmd)
    if status != 0:
        raise Exception('Command {} failed'.format(cmd))
    cmd = 'gcc -fPIC -shared -O{} ca_{}_hessian.c -o ca_{}_hessian.so'.format(opt, ident, ident)
    status = os.system(cmd)
    if status != 0:
        raise Exception('Command {} failed'.format(cmd))
