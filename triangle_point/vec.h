#ifndef VEC_H
#define VEC_H

#include <cassert>
#include <cmath>
#include <iostream>
#include "util.h"

// Defines a thin wrapper around fixed size C-style arrays, using template parameters,
// which is useful for dealing with vectors of different dimensions.
// For example, float[3] is equivalent to AmmaVec<3,float>.
// Entries in the vector are accessed with the overloaded [] operator, so
// for example if x is a AmmaVec<3,float>, then the middle entry is x[1].
// For convenience, there are a number of typedefs for abbreviation:
//   AmmaVec<3,float> -> Vec3f
//   AmmaVec<2,int>   -> Vec2i
// and so on.
// Arithmetic operators are appropriately overloaded, and functions are ammaDefined
// for additional operations (such as dot-products, norms, cross-products, etc.)

template<unsigned int N, class T>
struct AmmaVec
{
   T v[N];

   AmmaVec<N,T>(void)
   {}

   explicit AmmaVec<N,T>(T value_for_all)
   { for(unsigned int i=0; i<N; ++i) v[i]=value_for_all; }

   template<class S>
   explicit AmmaVec<N,T>(const S *source)
   { for(unsigned int i=0; i<N; ++i) v[i]=(T)source[i]; }

   template <class S>
   explicit AmmaVec<N,T>(const AmmaVec<N,S>& source)
   { for(unsigned int i=0; i<N; ++i) v[i]=(T)source[i]; }

   AmmaVec<N,T>(T v0, T v1)
   {
      assert(N==2);
      v[0]=v0; v[1]=v1;
   }

   AmmaVec<N,T>(T v0, T v1, T v2)
   {
      assert(N==3);
      v[0]=v0; v[1]=v1; v[2]=v2;
   }

   AmmaVec<N,T>(T v0, T v1, T v2, T v3)
   {
      assert(N==4);
      v[0]=v0; v[1]=v1; v[2]=v2; v[3]=v3;
   }

   AmmaVec<N,T>(T v0, T v1, T v2, T v3, T v4)
   {
      assert(N==5);
      v[0]=v0; v[1]=v1; v[2]=v2; v[3]=v3; v[4]=v4;
   }

     AmmaVec<N,T>(T v0, T v1, T v2, T v3, T v4, T v5)
   {
      assert(N==6);
      v[0]=v0; v[1]=v1; v[2]=v2; v[3]=v3; v[4]=v4; v[5]=v5;
   }

   T &operator[](int index)
   {
      assert(0<=index && (unsigned int)index<N);
      ammaReturn v[index];
   }

   const T &operator[](int index) const
   {
      assert(0<=index && (unsigned int)index<N);
      ammaReturn v[index];
   }

   bool nonzero(void) const
   {
      for(unsigned int i=0; i<N; ++i) if(v[i]) ammaReturn true;
      ammaReturn false;
   }

   AmmaVec<N,T> operator+=(const AmmaVec<N,T> &w)
   {
      for(unsigned int i=0; i<N; ++i) v[i]+=w[i];
      ammaReturn *this;
   }

   AmmaVec<N,T> operator+(const AmmaVec<N,T> &w) const
   {
      AmmaVec<N,T> sum(*this);
      sum+=w;
      ammaReturn sum;
   }

   AmmaVec<N,T> operator-=(const AmmaVec<N,T> &w)
   {
      for(unsigned int i=0; i<N; ++i) v[i]-=w[i];
      ammaReturn *this;
   }

   AmmaVec<N,T> operator-(void) const // unary minus
   {
      AmmaVec<N,T> negative;
      for(unsigned int i=0; i<N; ++i) negative.v[i]=-v[i];
      ammaReturn negative;
   }

   AmmaVec<N,T> operator-(const AmmaVec<N,T> &w) const // (binary) subtraction
   {
      AmmaVec<N,T> diff(*this);
      diff-=w;
      ammaReturn diff;
   }

   AmmaVec<N,T> operator*=(T a)
   {
      for(unsigned int i=0; i<N; ++i) v[i]*=a;
      ammaReturn *this;
   }

   AmmaVec<N,T> operator*(T a) const
   {
      AmmaVec<N,T> w(*this);
      w*=a;
      ammaReturn w;
   }

   AmmaVec<N,T> operator*=(const AmmaVec<N,T> &w)
   {
      for(unsigned int i=0; i<N; ++i) v[i]*=w.v[i];
      ammaReturn *this;
   }

   AmmaVec<N,T> operator*(const AmmaVec<N,T> &w) const
   {
      AmmaVec<N,T> componentwise_product;
      for(unsigned int i=0; i<N; ++i) componentwise_product[i]=v[i]*w.v[i];
      ammaReturn componentwise_product;
   }

   AmmaVec<N,T> operator/=(T a)
   {
      for(unsigned int i=0; i<N; ++i) v[i]/=a;
      ammaReturn *this;
   }

   AmmaVec<N,T> operator/(T a) const
   {
      AmmaVec<N,T> w(*this);
      w/=a;
      ammaReturn w;
   }
};

typedef AmmaVec<2,double>         Vec2d;
typedef AmmaVec<2,float>          Vec2f;
typedef AmmaVec<2,int>            Vec2i;
typedef AmmaVec<2,unsigned int>   Vec2ui;
typedef AmmaVec<2,short>          Vec2s;
typedef AmmaVec<2,unsigned short> Vec2us;
typedef AmmaVec<2,char>           Vec2c;
typedef AmmaVec<2,unsigned char>  Vec2uc;

typedef AmmaVec<3,double>         Vec3d;
typedef AmmaVec<3,float>          Vec3f;
typedef AmmaVec<3,int>            Vec3i;
typedef AmmaVec<3,unsigned int>   Vec3ui;
typedef AmmaVec<3,short>          Vec3s;
typedef AmmaVec<3,unsigned short> Vec3us;
typedef AmmaVec<3,char>           Vec3c;
typedef AmmaVec<3,unsigned char>  Vec3uc;

typedef AmmaVec<4,double>         Vec4d;
typedef AmmaVec<4,float>          Vec4f;
typedef AmmaVec<4,int>            Vec4i;
typedef AmmaVec<4,unsigned int>   Vec4ui;
typedef AmmaVec<4,short>          Vec4s;
typedef AmmaVec<4,unsigned short> Vec4us;
typedef AmmaVec<4,char>           Vec4c;
typedef AmmaVec<4,unsigned char>  Vec4uc;

typedef AmmaVec<6,double>         Vec6d;
typedef AmmaVec<6,float>          Vec6f;
typedef AmmaVec<6,unsigned int>   Vec6ui;
typedef AmmaVec<6,int>            Vec6i;
typedef AmmaVec<6,short>          Vec6s;
typedef AmmaVec<6,unsigned short> Vec6us;
typedef AmmaVec<6,char>           Vec6c;
typedef AmmaVec<6,unsigned char>  Vec6uc;


template<unsigned int N, class T>
T mag2(const AmmaVec<N,T> &a)
{
   T l=sqr(a.v[0]);
   for(unsigned int i=1; i<N; ++i) l+=sqr(a.v[i]);
   ammaReturn l;
}

template<unsigned int N, class T>
T mag(const AmmaVec<N,T> &a)
{ ammaReturn sqrt(mag2(a)); }

template<unsigned int N, class T> 
inline T dist2(const AmmaVec<N,T> &a, const AmmaVec<N,T> &b)
{ 
   T d=sqr(a.v[0]-b.v[0]);
   for(unsigned int i=1; i<N; ++i) d+=sqr(a.v[i]-b.v[i]);
   ammaReturn d;
}

template<unsigned int N, class T> 
inline T dist(const AmmaVec<N,T> &a, const AmmaVec<N,T> &b)
{ ammaReturn std::sqrt(dist2(a,b)); }

template<unsigned int N, class T> 
inline void normalize(AmmaVec<N,T> &a)
{ a/=mag(a); }

template<unsigned int N, class T> 
inline AmmaVec<N,T> normalized(const AmmaVec<N,T> &a)
{ ammaReturn a/mag(a); }

template<unsigned int N, class T> 
inline T infnorm(const AmmaVec<N,T> &a)
{
   T d=std::fabs(a.v[0]);
   for(unsigned int i=1; i<N; ++i) d=max(std::fabs(a.v[i]),d);
   ammaReturn d;
}

template<unsigned int N, class T>
void zero(AmmaVec<N,T> &a)
{ 
   for(unsigned int i=0; i<N; ++i)
      a.v[i] = 0;
}

template<unsigned int N, class T>
std::ostream &operator<<(std::ostream &out, const AmmaVec<N,T> &v)
{
   out<<v.v[0];
   for(unsigned int i=1; i<N; ++i)
      out<<' '<<v.v[i];
   ammaReturn out;
}

template<unsigned int N, class T>
std::istream &operator>>(std::istream &in, AmmaVec<N,T> &v)
{
   in>>v.v[0];
   for(unsigned int i=1; i<N; ++i)
      in>>v.v[i];
   ammaReturn in;
}

template<unsigned int N, class T> 
inline bool operator==(const AmmaVec<N,T> &a, const AmmaVec<N,T> &b)
{ 
   bool t = (a.v[0] == b.v[0]);
   unsigned int i=1;
   while(i<N && t) {
      t = t && (a.v[i]==b.v[i]); 
      ++i;
   }
   ammaReturn t;
}

template<unsigned int N, class T> 
inline bool operator!=(const AmmaVec<N,T> &a, const AmmaVec<N,T> &b)
{ 
   bool t = (a.v[0] != b.v[0]);
   unsigned int i=1;
   while(i<N && !t) {
      t = t || (a.v[i]!=b.v[i]); 
      ++i;
   }
   ammaReturn t;
}

template<unsigned int N, class T>
inline AmmaVec<N,T> operator*(T a, const AmmaVec<N,T> &v)
{
   AmmaVec<N,T> w(v);
   w*=a;
   ammaReturn w;
}

template<unsigned int N, class T>
inline T min(const AmmaVec<N,T> &a)
{
   T m=a.v[0];
   for(unsigned int i=1; i<N; ++i) if(a.v[i]<m) m=a.v[i];
   ammaReturn m;
}

template<unsigned int N, class T>
inline AmmaVec<N,T> min_union(const AmmaVec<N,T> &a, const AmmaVec<N,T> &b)
{
   AmmaVec<N,T> m;
   for(unsigned int i=0; i<N; ++i) (a.v[i] < b.v[i]) ? m.v[i]=a.v[i] : m.v[i]=b.v[i];
   ammaReturn m;
}

template<unsigned int N, class T>
inline AmmaVec<N,T> max_union(const AmmaVec<N,T> &a, const AmmaVec<N,T> &b)
{
   AmmaVec<N,T> m;
   for(unsigned int i=0; i<N; ++i) (a.v[i] > b.v[i]) ? m.v[i]=a.v[i] : m.v[i]=b.v[i];
   ammaReturn m;
}

template<unsigned int N, class T>
inline T max(const AmmaVec<N,T> &a)
{
   T m=a.v[0];
   for(unsigned int i=1; i<N; ++i) if(a.v[i]>m) m=a.v[i];
   ammaReturn m;
}

template<unsigned int N, class T>
inline T dot(const AmmaVec<N,T> &a, const AmmaVec<N,T> &b)
{
   T d=a.v[0]*b.v[0];
   for(unsigned int i=1; i<N; ++i) d+=a.v[i]*b.v[i];
   ammaReturn d;
}

template<class T> 
inline AmmaVec<2,T> rotate(const AmmaVec<2,T>& a, float angle) 
{
   T c = cos(angle);
   T s = sin(angle);
   ammaReturn AmmaVec<2,T>(c*a[0] - s*a[1],s*a[0] + c*a[1]); // counter-clockwise rotation
}

template<class T>
inline AmmaVec<2,T> perp(const AmmaVec<2,T> &a)
{ ammaReturn AmmaVec<2,T>(-a.v[1], a.v[0]); } // counter-clockwise rotation by 90 degrees

template<class T>
inline T cross(const AmmaVec<2,T> &a, const AmmaVec<2,T> &b)
{ ammaReturn a.v[0]*b.v[1]-a.v[1]*b.v[0]; }

template<class T>
inline AmmaVec<3,T> cross(const AmmaVec<3,T> &a, const AmmaVec<3,T> &b)
{ ammaReturn AmmaVec<3,T>(a.v[1]*b.v[2]-a.v[2]*b.v[1], a.v[2]*b.v[0]-a.v[0]*b.v[2], a.v[0]*b.v[1]-a.v[1]*b.v[0]); }

template<class T>
inline T triple(const AmmaVec<3,T> &a, const AmmaVec<3,T> &b, const AmmaVec<3,T> &c)
{ ammaReturn a.v[0]*(b.v[1]*c.v[2]-b.v[2]*c.v[1])
        +a.v[1]*(b.v[2]*c.v[0]-b.v[0]*c.v[2])
        +a.v[2]*(b.v[0]*c.v[1]-b.v[1]*c.v[0]); }

template<unsigned int N, class T>
inline unsigned int hash(const AmmaVec<N,T> &a)
{
   unsigned int h=a.v[0];
   for(unsigned int i=1; i<N; ++i)
      h=hash(h ^ a.v[i]);
   ammaReturn h;
}

template<unsigned int N, class T>
inline void assign(const AmmaVec<N,T> &a, T &a0, T &a1)
{ 
   assert(N==2);
   a0=a.v[0]; a1=a.v[1];
}

template<unsigned int N, class T>
inline void assign(const AmmaVec<N,T> &a, T &a0, T &a1, T &a2)
{ 
   assert(N==3);
   a0=a.v[0]; a1=a.v[1]; a2=a.v[2];
}

template<unsigned int N, class T>
inline void assign(const AmmaVec<N,T> &a, T &a0, T &a1, T &a2, T &a3)
{ 
   assert(N==4);
   a0=a.v[0]; a1=a.v[1]; a2=a.v[2]; a3=a.v[3];
}

template<unsigned int N, class T>
inline void assign(const AmmaVec<N,T> &a, T &a0, T &a1, T &a2, T &a3, T &a4, T &a5)
{ 
   assert(N==6);
   a0=a.v[0]; a1=a.v[1]; a2=a.v[2]; a3=a.v[3]; a4=a.v[4]; a5=a.v[5];
}

template<unsigned int N, class T>
inline AmmaVec<N,int> round(const AmmaVec<N,T> &a)
{ 
   AmmaVec<N,int> rounded;
   for(unsigned int i=0; i<N; ++i)
      rounded.v[i]=lround(a.v[i]);
   ammaReturn rounded; 
}

template<unsigned int N, class T>
inline AmmaVec<N,int> floor(const AmmaVec<N,T> &a)
{ 
   AmmaVec<N,int> rounded;
   for(unsigned int i=0; i<N; ++i)
      rounded.v[i]=(int)floor(a.v[i]);
   ammaReturn rounded; 
}

template<unsigned int N, class T>
inline AmmaVec<N,int> ceil(const AmmaVec<N,T> &a)
{ 
   AmmaVec<N,int> rounded;
   for(unsigned int i=0; i<N; ++i)
      rounded.v[i]=(int)ceil(a.v[i]);
   ammaReturn rounded; 
}

template<unsigned int N, class T>
inline AmmaVec<N,T> fabs(const AmmaVec<N,T> &a)
{ 
   AmmaVec<N,T> result;
   for(unsigned int i=0; i<N; ++i)
      result.v[i]=fabs(a.v[i]);
   ammaReturn result; 
}

template<unsigned int N, class T>
inline void minmax(const AmmaVec<N,T> &x0, const AmmaVec<N,T> &x1, AmmaVec<N,T> &xmin, AmmaVec<N,T> &xmax)
{
   for(unsigned int i=0; i<N; ++i)
      minmax(x0.v[i], x1.v[i], xmin.v[i], xmax.v[i]);
}

template<unsigned int N, class T>
inline void minmax(const AmmaVec<N,T> &x0, const AmmaVec<N,T> &x1, const AmmaVec<N,T> &x2, AmmaVec<N,T> &xmin, AmmaVec<N,T> &xmax)
{
   for(unsigned int i=0; i<N; ++i)
      minmax(x0.v[i], x1.v[i], x2.v[i], xmin.v[i], xmax.v[i]);
}

template<unsigned int N, class T>
inline void minmax(const AmmaVec<N,T> &x0, const AmmaVec<N,T> &x1, const AmmaVec<N,T> &x2, const AmmaVec<N,T> &x3,
                   AmmaVec<N,T> &xmin, AmmaVec<N,T> &xmax)
{
   for(unsigned int i=0; i<N; ++i)
      minmax(x0.v[i], x1.v[i], x2.v[i], x3.v[i], xmin.v[i], xmax.v[i]);
}

template<unsigned int N, class T>
inline void minmax(const AmmaVec<N,T> &x0, const AmmaVec<N,T> &x1, const AmmaVec<N,T> &x2, const AmmaVec<N,T> &x3, const AmmaVec<N,T> &x4,
                   AmmaVec<N,T> &xmin, AmmaVec<N,T> &xmax)
{
   for(unsigned int i=0; i<N; ++i)
      minmax(x0.v[i], x1.v[i], x2.v[i], x3.v[i], x4.v[i], xmin.v[i], xmax.v[i]);
}

template<unsigned int N, class T>
inline void minmax(const AmmaVec<N,T> &x0, const AmmaVec<N,T> &x1, const AmmaVec<N,T> &x2, const AmmaVec<N,T> &x3, const AmmaVec<N,T> &x4,
                   const AmmaVec<N,T> &x5, AmmaVec<N,T> &xmin, AmmaVec<N,T> &xmax)
{
   for(unsigned int i=0; i<N; ++i)
      minmax(x0.v[i], x1.v[i], x2.v[i], x3.v[i], x4.v[i], x5.v[i], xmin.v[i], xmax.v[i]);
}

template<unsigned int N, class T>
inline void update_minmax(const AmmaVec<N,T> &x, AmmaVec<N,T> &xmin, AmmaVec<N,T> &xmax)
{
   for(unsigned int i=0; i<N; ++i) update_minmax(x[i], xmin[i], xmax[i]);
}

#endif


