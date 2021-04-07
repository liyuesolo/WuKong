#ifndef EOL_NODE_H
#define EOL_NODE_H

#include "VecMatDef.h"

template<class T>
class EoLNode
{
    
private:
    TV3 x;
    TV2 u;
    int idx;
    bool is_crossing;
    T radius;
    std::vector<EoLNode*> neighbors;

public:

    void attachNeigbor(const EoLNode* neighbor)
    {
        neighbors.push_back(neighbor);
    }

    EoLNode(TV3 pos3d, TV3 pos2d, int _idx, bool _is_crossing, T _radius) : x(pos3d), u(pos2d), idx(_idx), is_crossing(_is_crossing), radius(_radius) {}
    
    EoLNode() {}
    ~EoLNode() {}
};

#endif