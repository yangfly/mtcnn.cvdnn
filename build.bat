:: set VS 2017 or 2015 
@set VS=2017
@set build_type=Release

@if "%VS%" == "2017" (@call :set_vs2017) else (@call :set_vs2015)

@if exist build (@rmdir /S /Q build)
@mkdir build
@cd build
cmake ../src -G%generator% -DCMAKE_CONFIGURATION_TYPES=%build_type% -DSUB_FLOBER=%folder%

@for %%t in (%build_type%) do (@call :copy_dll %%t %folder%)
@pause

:set_vs2017
  @set generator="Visual Studio 15 2017 Win64"
  @set folder=vc15
  @goto :eof

:set_vs2015
  @set generator="Visual Studio 14 2015 Win64"
  @set folder=vc14
  @goto :eof

:copy_dll
  @set type=%1
  @set folder_=%2
  @mkdir %type%
  @copy %cd%\..\opencv\%folder_%\bin\*.dll %type%\
  @goto :eof
