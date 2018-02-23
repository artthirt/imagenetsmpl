#ifndef IMNET_LIST_H
#define IMNET_LIST_H

#include <string>

namespace imnet{

extern std::string imnet_list[];
extern size_t count_imnet_list;

int getNumberOfList(const std::string& name);

}

#endif // IMNET_LIST_H
