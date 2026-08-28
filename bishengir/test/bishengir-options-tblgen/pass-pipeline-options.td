// RUN: bishengir-options-tblgen -gen-pass-options %s -I %p/../../include/ | FileCheck %s --check-prefix=PASS
// RUN: bishengir-options-tblgen -gen-pass-pipeline-options %s -I %p/../../include/ | FileCheck %s --check-prefix=PIPELINE
// RUN: bishengir-options-tblgen -gen-configs-and-options-conversion %s -I %p/../../include/ | FileCheck %s --check-prefix=SET-UP

include "bishengir/Tools/ConfigOptions/Options.td"

def Foo : Option {
  let description = "Foooooo.";
  let type = "int64_t";
  let defaultValue = "1";
  let passGroup = [];
  // Additional strings will be directly emitted
  let additionalOptFlags = [{
    some random string
  }];
}

def Bar : Option {
  let type = "bool";
  let defaultValue = "false";
  let passGroup = ["DIALECT_A"];
}

def List : ListOption {
  let type = "std::string";
  let passGroup = ["DIALECT_B"];
}

// PASS: #ifdef GEN_DIALECT_B_OPTION_REGISTRATION

// PASS: Pass::ListOption<std::string>
// PIPELINE: PassOptions::ListOption<std::string>

// PASS: list{
// PASS:   *this
// PASS:   ,"list"
// PASS:   ,llvm::cl::desc("")
// PASS: };
// PASS: #undef GEN_DIALECT_B_OPTION_REGISTRATION
// PASS: #endif // GEN_DIALECT_B_OPTION_REGISTRATION

// Every options will appear under `GEN_ALL_OPTION_REGISTRATION`

// PASS: #ifdef GEN_ALL_OPTION_REGISTRATION

// PASS: Pass::Option<bool>
// PIPELINE: PassOptions::Option<bool>

// PASS: bar{
// PASS:   *this
// PASS:   ,"bar"
// PASS:   ,llvm::cl::desc("")
// PASS:   ,llvm::cl::init(false)
// PASS: };

// PASS: Pass::Option<int64_t>
// PIPELINE: PassOptions::Option<int64_t>

// PASS: foo{
// PASS:   *this
// PASS:   ,"foo"
// PASS:   ,llvm::cl::desc("Foooooo.")
// PASS:   ,llvm::cl::init(1)
// PASS:     ,
// PASS:     some random string
// PASS: };

// PASS: Pass::ListOption<std::string>
// PIPELINE: PassOptions::ListOption<std::string>

// PASS: list{
// PASS:   *this
// PASS:   ,"list"
// PASS:   ,llvm::cl::desc("")
// PASS: };
// PASS: #undef GEN_ALL_OPTION_REGISTRATION
// PASS: #endif // GEN_ALL_OPTION_REGISTRATION

// PASS: #ifdef GEN_DIALECT_A_OPTION_REGISTRATION

// PASS: Pass::Option<bool>
// PIPELINE: PassOptions::Option<bool>

// PASS: bar{
// PASS:   *this
// PASS:   ,"bar"
// PASS:   ,llvm::cl::desc("")
// PASS:   ,llvm::cl::init(false)
// PASS: };
// PASS: #undef GEN_DIALECT_A_OPTION_REGISTRATION
// PASS: #endif // GEN_DIALECT_A_OPTION_REGISTRATION

// SET-UP: #ifdef GEN_DIALECT_B_OPTION_SETUP
// SET-UP: options.list = config.getList();
// SET-UP: #undef GEN_DIALECT_B_OPTION_SETUP
// SET-UP: #endif // GEN_DIALECT_B_OPTION_SETUP

// SET-UP: #ifdef GEN_DIALECT_A_OPTION_SETUP
// SET-UP: options.bar = config.getBar();
// SET-UP: #undef GEN_DIALECT_A_OPTION_SETUP
// SET-UP: #endif // GEN_DIALECT_A_OPTION_SETUP
