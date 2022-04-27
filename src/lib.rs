use pgx::*;
use serde::{Deserialize, Serialize};
use std::fmt::{Display, Debug, Error, Formatter};
use std::result::Result;
use std::hash::{Hasher, Hash};
extern crate blas;

//mod vector_search;

pg_module_magic!();

#[derive(
    //Eq,
    PartialEq,
    //Ord,
    Hash,
    //PartialOrd,
    PostgresType,
    Serialize,
    Deserialize,
    PostgresEq,
    //PostgresOrd,
    //PostgresHash,
)]
#[inoutfuncs]
pub struct Vector(Vec<f32>);

impl Display for Vector {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), Error> {
        self.0.iter().take(3).fold(write!(f, "["), |result, d| {
            result.and_then(|_| write!(f, "{}, ", d))
        }).and_then(|_| write!(f, "...]"))
    }
}

impl Debug for Vector {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), Error> {
        self.0.iter().fold(write!(f, "["), |result, d| {
            result.and_then(|_| write!(f, "{},", d))
        }).and_then(|_| write!(f, "...]"))
    }
}

impl InOutFuncs for Vector {
    fn input(input: &pgx::cstr_core::CStr) -> Self
    where
        Self: Sized,
    {
        let string = input.to_str().expect("invalid UTF-8");
        let arr = string[1..string.len() - 1].split(",").map(|s| { s.trim().parse::<f32>().unwrap() }).collect();

        Vector(arr)
    }

    fn output(&self, buffer: &mut StringInfo) {
        buffer.push_str(&format!("{}", self))
    }
}

impl Hash for Vector {
    fn hash<H: Hasher>(&self, state: &mut H) {
        std::slice::from_raw_parts(self.0.as_ptr(), vec.len() * 4).hash(state);
    }
}

#[pg_operator(immutable, parallel_safe)]
#[opname(<->)]
fn l2_distance(v1: Vector, v2: Vector) -> f32 {
    let a = v1.0;
    let b = v2.0;
     // TODO handling of NaN and stuff like this
     if a.len() != b.len() {
        panic!("the vectors dimension mismatch {} != {}", a.len(), b.len());
    }

    // c = b.clone() does not work here because cblas_daxpy
    // modifies the content of c and cloned() on a slice does
    // not create a copy.
    let mut c: Vec<f32> = b.to_vec();

    unsafe {
        blas::saxpy(
            a.len() as i32,
            -1.0,
            &a,
            1,
            &mut c,
            1
        );

        blas::snrm2(
            c.len() as i32,
            &c,
            1
        )
    }
}

#[cfg(any(test, feature = "pg_test"))]
#[pg_schema]
mod tests {
    use pgx::*;

    #[pg_test]
    fn test_str2vec() {
        assert_eq!(crate::Vector(vec![1.,2.,3., 4., 5.]), 
            crate::Vector::input(
                pgx::cstr_core::CStr::from_bytes_with_nul(b"[1., 2., 3., 4., 5.]\0").expect("CStr::from_bytes_with_nul failed")));
    }

}

#[cfg(test)]
pub mod pg_test {
    pub fn setup(_options: Vec<&str>) {
        // perform one-off initialization when the pg_test framework starts
    }

    pub fn postgresql_conf_options() -> Vec<&'static str> {
        // return any postgresql.conf settings that are required for your tests
        vec![]
    }
}
