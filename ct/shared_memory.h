#ifndef SHARED_MEMORY_H
#define SHARED_MEMORY_H

#include <vector>
#include <iostream>
#include <sstream>
#include <string>

namespace sm{

template< class T >
class shared_memory
{
public:
	shared_memory(){
		m_val = nullptr;
		m_ref = nullptr;
	}
	shared_memory(const shared_memory<T>& val){
		m_val = val.m_val;
		m_ref = val.m_ref;
		if(m_ref)
			(*m_ref)++;
	}
	~shared_memory(){
		if(m_ref){
			(*m_ref)--;
			if(*m_ref <= 0 && m_val){
				delete m_val;
				delete m_ref;
			}
		}
	}

	shared_memory<T>& operator=(const shared_memory<T>& val){
		if(m_val && m_ref){
			--(*m_ref);
			if(*m_ref <= 0 && m_val != val.m_val){
				delete m_val;
				delete m_ref;
			}
			if(m_val == val.m_val){
				//(*m_ref)++;
			}else{
				m_ref = nullptr;
			}
		}

		m_val = val.m_val;
		m_ref = val.m_ref;
		if(m_ref)
			(*m_ref)++;
		return *this;
	}
	void release(){
		if(m_val && m_ref){
			--(*m_ref);
			if(*m_ref <= 0){
				delete m_val;
				delete m_ref;

			}
			m_val = nullptr;
			m_ref = nullptr;
		}
	}

	bool empty() const{
		return m_val == nullptr;
	}
	inline T* get(){
		return m_val;
	}
	inline T* get() const{
		return m_val;
	}
	inline T& operator*(){
		return *m_val;
	}
	inline T& operator*() const{
		return *m_val;
	}
	inline int ref() const{
		if(m_ref)
			return *m_ref;
		return -1;
	}
	inline T& operator() (){
		return *m_val;
	}

	template< class N >
	friend shared_memory< N > make_shared();

private:
	T* m_val;
	int *m_ref;

	shared_memory(T* val){
		m_val = nullptr;
		m_ref = nullptr;
		if(val){
			m_ref = new int(1);
			m_val = val;
		}
	}
};

template< class T >
shared_memory< T > make_shared()
{
	shared_memory< T > res(new T());

	return res;
}

}

#endif // SHARED_MEMORY_H
