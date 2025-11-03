// App headers
#include <devel/init.hh>
// Project Headers
#include <protocols/moves/Mover.hh>
#include <core/conformation/membrane/Span.hh>
#include <core/conformation/membrane/SpanningTopology.hh>
#include <protocols/membrane/util.hh>

#include <core/conformation/Residue.hh>
#include <core/chemical/ResidueType.hh>
#include <core/chemical/ResidueTypeSet.hh>
#include <core/chemical/ChemicalManager.hh>
#include <core/pack/task/TaskFactory.hh>
#include <core/pack/pack_rotamers.hh>

#include <basic/options/option.hh>
#include <basic/options/keys/mp.OptionKeys.gen.hh>
#include <basic/options/keys/in.OptionKeys.gen.hh>
#include <basic/options/keys/out.OptionKeys.gen.hh>
#include <basic/options/keys/relax.OptionKeys.gen.hh>

#include <core/sequence/Sequence.hh>
#include <core/sequence/util.hh>
#include <core/pose/annotated_sequence.hh>
#include <protocols/abinitio/ClassicAbinitio.hh>
#include <protocols/simple_moves/GunnCost.hh>
#include <protocols/simple_moves/SymmetricFragmentMover.hh>
#include <protocols/simple_moves/SwitchResidueTypeSetMover.hh>
#include <protocols/simple_moves/BackboneMover.hh>
#include <protocols/simple_moves/SuperimposeMover.hh>
#include <protocols/simple_moves/FragmentMover.fwd.hh>
#include <core/kinematics/MoveMap.hh>
#include <core/kinematics/FoldTree.hh>

#include <core/scoring/ScoreFunctionFactory.hh>
#include <core/scoring/ScoreFunction.hh>
#include <core/scoring/Energies.hh>
#include <core/scoring/constraints/Constraint.hh>
#include <core/scoring/constraints/AtomPairConstraint.hh>
#include <core/scoring/constraints/DihedralConstraint.hh>
#include <core/scoring/constraints/AngleConstraint.hh>
#include <core/scoring/func/SplineFunc.hh>

#include <core/optimization/MinimizerOptions.hh>
#include <core/optimization/AtomTreeMinimizer.hh>

#include <protocols/moves/MonteCarlo.fwd.hh>
#include <protocols/moves/MonteCarlo.hh>
#include <protocols/simple_moves/FragmentMover.hh>
#include <core/fragment/FragSet.hh>
#include <core/fragment/FragmentIO.hh>

#include <protocols/trRosetta/trRosettaOutputs_v1.hh>
#include <protocols/trRosetta/trRosettaProtocol_v1.hh>
#include <protocols/trRosetta_protocols/movers/trRosettaProtocolMover.hh>
#include <protocols/trRosetta_protocols/movers/trRosettaProtocolMoverCreator.hh>

#include <core/chemical/AA.hh>
#include <core/select/residue_selector/ResidueIndexSelector.hh>
#include <core/scoring/RamaPrePro.hh>
#include <core/simple_metrics/metrics/RMSDMetric.hh>
#include <core/simple_metrics/metrics/TotalEnergyMetric.hh>
#include <core/simple_metrics/metrics/TimingProfileMetric.hh>
#include <core/util/SwitchResidueTypeSet.hh>

#include <protocols/trRosetta_protocols/constraint_generators/trRosettaConstraintGenerator.hh>
#include <protocols/simple_moves/MutateResidue.hh>
#include <protocols/residue_selectors/StoredResidueSubsetSelector.hh>
#include <protocols/residue_selectors/StoreResidueSubsetMover.hh>
#include <protocols/simple_moves/bin_transitions/InitializeByBins.hh>
#include <protocols/minimization_packing/MinMover.hh>
#include <protocols/moves/RepeatMover.hh>
#include <protocols/moves/MoverContainer.hh>
#include <protocols/moves/TrialMover.hh>
#include <protocols/moves/WhileMover.hh>
#include <protocols/relax/FastRelax.hh>
#include <protocols/moves/CompositionMover.hh>
// Package Headers
#include <apps/benchmark/performance/init_util.hh>
#include <core/types.hh>
#include <numeric/xyzVector.hh>
#include <numeric/constants.hh>
#include <core/id/AtomID.hh>
#include <core/pose/Pose.hh>
#include <core/pose/PDBInfo.hh>
#include <core/import_pose/import_pose.hh>
#include <core/pose/util.hh>
#include <core/pose/variant_util.hh>
#include <core/pose/subpose_manipulation_util.hh>
#include <core/conformation/util.hh>
#include <protocols/jd2/JobDistributor.hh>
#include <protocols/jd2/Job.hh>
#include <basic/Tracer.hh>

// utility headers
#include <utility/excn/Exceptions.hh>
#include <utility/excn/Exceptions.hh>
#include <utility/string_util.hh>
#include <utility/io/ozstream.hh>
#include <utility/string_util.hh>
#include <utility/minmax.hh>
#include <numeric/random/random.hh>
#include <protocols/jd2/util.hh>
#include <core/scoring/rms_util.hh>
#include <core/scoring/rms_util.tmpl.hh>
#include <basic/options/keys/score.OptionKeys.gen.hh>
#include <basic/options/keys/trRosetta.OptionKeys.gen.hh>
#include <basic/tensorflow_manager/util.hh>
#include <utility/tag/Tag.hh>
#include <utility/pointer/memory.hh>

// C++ Headers
#include <cstdlib>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <utility>
#include <algorithm>
#include <map>
#include <iomanip>
#include <stdlib.h>
#include <math.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include "pareto.hh"
#include "ClassicAbinitio.hh"

using core::Real;
using namespace std;
using namespace core;
using namespace core::pose;
using namespace core::conformation;
using namespace basic::options;
using namespace basic::options::OptionKeys;
using namespace protocols::simple_moves;
using namespace core::chemical;
using namespace protocols::moves;
using namespace core::scoring;
using namespace protocols::trRosetta;
using namespace protocols::trRosetta_protocols::movers;
using namespace core::scoring::constraints;
using namespace core::scoring::func;

ofstream processinformation;

static basic::Tracer TR( "apps.public.multidomain_assembly" );


class DomainAssembly : public Mover{

public:

	//////////////////////////
	/// @note Constructors ///
	//////////////////////////

	/// @brief Construct a Default Mover
	DomainAssembly();

	/// @brief Copy Constructor
	/// @details Make a deep copy of this mover object
	DomainAssembly( DomainAssembly const & src );

	/// @brief Assignment Operator
	/// @details Make a deep copy of this mover object, overriding the assignment operator
	DomainAssembly &
	operator=( DomainAssembly const & src );

	/// @brief Destructor
	~DomainAssembly() override;

	///////////////////////////
	/// @note Mover Methods ///
	///////////////////////////

	/// @brief Get the name of this mover
	std::string get_name() const override;
    Real Iterate = 0;


	/// @brief Get protein interface statistics
	void apply( Pose & pose ) override;
    //**********************************************************
	struct DistanceType
	{
		Size re1;
		Size re2;
		Real dist;
		Real confidence;

		DistanceType()
		{
			re1 = 0;
			re2 = 0;
			dist = 0;
			confidence = 0;
		}
	};

	utility::vector1<DistanceType> vec_distance1;




private: // parameter

	/// @brief Maximum number of generations (angle)
	Size G = 20; //300

	/// @brief Population size (pose)
	Size NP1 = 5;  //5

	/// @brief Population size (angle)
	Size NP2 = 100;  //100

	/// @brief Maximum disturbance of initial angle
	Real max_disturbance = 0.5;

	/// @brief Scaling factor
	Real F = 0.5;

	/// @brief Crossover factor
	Real CR = 0.5;

	/// @brief Number of candidate individual
	Size N_candidate = 7;  //100

	/// @brief Number of output model
	Size N_output = 5;

	/// @brief Lenth of linker
	Size L_linker = 8;

	/// @brief Will this generate distance constraints?
	bool set_generate_dist_constraints_ = true;

	/// @brief Probability cutoff for distance constraints.  Default 0.05.
	core::Real dist_prob_cutoff_ = 0.05;

	/// @brief Probability cutoff for distance constraints in relax.  Default 0.15.
	core::Real ref_dist_prob_cutoff_ = 0.15;

	/// @brief Constraint weight for distance constraints.  Default 1.0.
	core::Real distance_constraint_weight_ = 1.0;

	/// @brief Number of dist bins
	Size const n_dist_bins_ = 37;

	/// @brief Does relax begin?
	bool relax_ = true;


private: // route

	std::string dist_path1_ = "./AF3_inter.txt";
	std::string dist_path2_ = "./distprob.txt";

	std::string index_path_ = "./index.txt";

	std::string output_path_ = "./trans_mat/trans_mat.txt";

private: // data

    /// @brief Length of the full-length protein
	Size nres;
	/// @brief Length of the antibody-length protein
	Size nres_ab;

	/// @brief Starting and ending indexes of the CDR
	utility::vector1<Size> cdr_starts;
    utility::vector1<Size> cdr_ends;


	/// @brief Sequence of the full-length protein: given as fasta
	std::string full_seq_;

	utility::vector1<utility::vector1<Size>> every_cluster;
	utility::vector1<core::pose::Pose > seed_population;

	utility::vector1<core::pose::Pose > all_generate_pose;


	/// @brief Native pose
//	Pose native;

	/// @brief Number of the pose in the -in:file:l list that is the TM domain
	Size tmpdb_{};

	/// @brief Vector of pose input files
	utility::vector1< std::string > infiles_;

	/// @brief Vector of poses_ that are read in from PDB files
	utility::vector1< core::pose::Pose > poses_;

	utility::vector1< core::pose::Pose > chains_;

	/// @brief Vector of sequences_ in poses_
	utility::vector1< std::string > sequences_;

	/// @brief Vector of offsets_, i.e. residue numbers along the full-length sequence
	///   for which the PDB poses_ start
	utility::vector1< Size > offsets_;

	/// @brief Vector of linker sequences_ to model
	utility::vector1< std::string > linkers_;

	/// @brief Vector of index of start and end of movable residues
	utility::vector1< std::pair< Size, Size > > movable_residues_;

	/// @brief Vector of index of start and end of immovable residues
	utility::vector1< std::pair< Size, Size > > immovable_residues_;

	/// @brief Movemap of which residues can move (i.e. are in linkers_ +/-1 residue)
	core::kinematics::MoveMap movemap_;

	/// @brief score function
	core::scoring::ScoreFunctionOP sfxn_;

	/// @brief full-atom score function
	core::scoring::ScoreFunctionOP sfxn_fullatom_;

	/// @brief The dist constraints from the neural network
	utility::vector1< utility::vector1< utility::vector1< Real > > > dist_constraints;

	/// @brief Affine transformations from the neural network
	utility::vector1< utility::vector1< utility::vector1< utility::vector1< Real > > > > affine_transformations;

	/// @brief save score of pose
	struct SCORE
	{
		Real total_score;
		Real rmsd;

		SCORE() { total_score = 0.0; rmsd = 0.0; }
		SCORE( Real v1, Real v2 ) : total_score(v1), rmsd(v2) {}
	};

	/// @brief population of full pose
	utility::vector1< Pose > population_pose_;
	utility::vector1< Pose > rot_pose;

	utility::vector1< utility::vector1< utility::vector1< Real > > > population_rot_tra_;

	/// @brief population of pose score
    utility::vector1< Real > population_score_;
	utility::vector1< SCORE > population_score1_;
	utility::vector1< Real > population_score2_;

	/// @brief rotation axis
	utility::vector1< std::pair< numeric::xyzVector< Real >, numeric::xyzVector< Real > > > rotation_axis_;

	/// @brief rotation point
	utility::vector1< numeric::xyzVector< Real > > rotation_point_;

	/// @brief sort by total score
	multimap< Real, Pose > sort_total_score_pose_;

	/// @brief relaxed model
	multimap< Real, Pose > sort_ref_total_score_pose_;
	multimap< Real, Pose > sort_ref_total_score_pose_gen;

//	utility::vector1<utility::vector1<Real>> dist_constr1_;
//    utility::vector1<utility::vector1<Real>> dist_constr2_;
    utility::vector1<utility::vector1<utility::vector1<Real>>> dist_constr1_;
    utility::vector1<utility::vector1<utility::vector1<Real>>> dist_constr2_;
    utility::vector1<utility::vector1<utility::vector1<Real>>> dist_constr3_;
    utility::vector1<utility::vector1<utility::vector1<Real>>> dist_constr4_;
//    protocols::moves::MonteCarloOP mc_;
//    protocols::moves::MonteCarloOP mc_ptr();
////////////////////////////////////////////
//用于处理蛋白质结构建模和预测任务中的各种操作。用于优化构象、
//计算得分、生成约束等，以支持蛋白质结构的建模和分析
////////////////////////////////////////////
private: // methods

    //**********************************************************
	//add methods
	void Map_matrix(vector<vector<double>> Map_matrix_pose);
	void Kmediods(Size K, vector<vector<double>> Distance_Matrix);
	double DM_score(core::pose::Pose &pose, core::pose::Pose &tempPose);
	void DMscore_and_cluster(utility::vector1<core::pose::Pose> &storage_pose);
	void cluster_and_sort(Size &choose_num, utility::vector1<core::pose::Pose> &target_population);
	void delete_unwanted_pose(utility::vector1<core::pose::Pose> &target_population, utility::vector1<Size> &target_index);
	utility::vector1<utility::vector1<utility::vector1<Real>>> delete_unwanted_rot_tra(utility::vector1< utility::vector1< utility::vector1< Real > > > target_rot_tra, utility::vector1<Size> &target_index);
	void Pareto_method1(utility::vector1<Size> &target_index, utility::vector1<core::pose::Pose> &target_population, utility::vector1<Real> &population_energy, utility::vector1< core::pose::Pose > poses_);
	void Pareto_method2(utility::vector1<Size> &target_index, utility::vector1< utility::vector1< utility::vector1< Real > > > target_rot_tra, utility::vector1<Real> &population_score_);
	void Read_distance();
    //**********************************************************

	/// @brief Register Options with JD2
	void register_options();

	/// @brief Initialize Mover options from the comandline
	void init_from_cmd();

	/// @brief create residue
	ResidueOP create_residue_from_olc( char olc );

	/// @brief create residue
	Residue create_residue_from_resn( Pose & pose, Size resnumber );

	/// @brief set random torsion
	void set_random_torsion( Pose & pose, Size resn );

	utility::vector1< utility::vector1< Real > > set_random_rot_tra();

	/// @brief Print score to cout
	void print_score( Pose & pose, core::scoring::ScoreFunctionOP sfxn );

	/// @brief calculate RMSD of pose
	Real cal_rmsd( Pose & pose );

	/// @brief calculate total score of pose
	Real cal_total_score( Pose & pose );

	/// @brief calculate ref total score of pose
	Real cal_ref_total_score( Pose & pose );

	/// @brief calculate distance score of pose
	Real cal_Cscore( Pose & pose , utility::vector1< core::pose::Pose > poses_);
    Real cal_Cscore_antibody( Pose & pose );
    /// @brief defines reference frame per residue using backbone atoms
    utility::vector1< utility::vector1< Real > > set_lframe( Pose & pose, Size res );

    //**********************************************************
//	utility::vector1< DistanceType > vec_distance1;  //创建了一个名为 vec_distance1 的空向量（vector_distance1)
    //**********************************************************

    /// @brief calculate frame aligned point error (CA)
//    Real frame_aligned_point_error_CA( Pose & pose );

    Real frame_aligned_point_error_CA_CDR_v1( Pose & pose, utility::vector1< core::pose::Pose > poses_);
    Real frame_aligned_point_error_CA_CDR_v2( Pose & pose, utility::vector1< core::pose::Pose > poses_);
	Real score_rot_tra_v3( utility::vector1< utility::vector1< Real > > rot_tra_ );

    Real score_rot_tra_v4( utility::vector1< utility::vector1< Real > > rot_tra_ );

	utility::vector1< utility::vector1< Real > > EulerAngles_to_rotationMatrix( Real euler_x, Real euler_y, Real euler_z );

    utility::vector1< Real > spherical_to_translationVector( Real spher_r, Real spher_t, Real spher_p );

	utility::vector1< utility::vector1< Real > > matrix_multiply( utility::vector1< utility::vector1< Real > > arrA, utility::vector1< utility::vector1< Real > > arrB );

    /// @brief calculate frame aligned point error (bb)
    // Real frame_aligned_point_error_bb( Pose & pose );

	/// @brief Transform the index of residue number (full to part)
	Size index_f2p( Size f_index );

	/// @brief Transform the index of residue number (part to full)
	Size index_p2f( bool is_axis, Size p_index );

	/// @brief Accepted with Boltzmann probability
	bool boltzmann_accept( const Real targetEnergy, const Real trialEnergy, Real recipocal_KT );
	bool greedy_accept( const Real targetEnergy, const Real trialEnergy );
	/// @brief Score of rotation angle
//	Real score_rotation_angle( utility::vector1< Real > & rotation_angle );

	/// @brief Score of rotation angle
	std::pair<Pose, Real> score_rotation_angle_v1( Pose & pose, utility::vector1< Real > & rotation_angle, utility::vector1< core::pose::Pose > poses_ );


    //**********************************************************
    Real Distance_score2(core::pose::Pose &pose, utility::vector1<DistanceType> &vec_distance);
    //**********************************************************


//	Pose generate_pose( Pose & pose, utility::vector1< Real > & rotation_angle );

	/// @brief Solving the rotation angle by differential evolution algorithm
	std::pair<utility::vector1<Real>,utility::vector1<core::pose::Pose>> rotation_angle_optimization( Size index ,utility::vector1< core::pose::Pose > poses_);

	/// @brief Given a pose, generate the constraints
	core::scoring::constraints::ConstraintCOPs generate_constraints( Pose const & pose ) const;
	
	/// @brief Generate the distance constraints
	void generate_dist_constraints(
		utility::vector1 <core::scoring::constraints::ConstraintCOP> & outputvec,
		utility::vector1< Real > const & dist_bins_background,
		utility::vector1< Real > const & dist_bins_vect,
		core::id::AtomID const & cb_atom_i,
		core::id::AtomID const & cb_atom_j,
		Size const n_dist_bins,
		Size const n_dist_bins_model,
		Size const ires,
		Size const jres,
		Real const prob_cutoff
	) const;

	/// @brief Remove previously-added constraints from the pose
	void remove_constraints(
		utility::vector1< core::scoring::constraints::ConstraintCOP > const & constraints,
		Pose & pose
	) const;

	/// @brief Add constraints from a list to the pose, if the constraints are between residues that are
	/// separated by at least min_seqsep but less than max_seqsep residues.
	/// @details This does not clear constraints from the pose.
	/// @note It is "less than" max_seqsep, not "less than or equal to".
	void add_constraints_to_pose(
		Pose & pose,
		utility::vector1< core::scoring::constraints::ConstraintCOP > const & constraints,
		Size const min_seqsep,
		Size const max_seqsep,
		bool const skip_glycine_positions = false
	) const;

	/// @brief Given a constraint, determine if it is an AtomPairConstraint, an
	/// AngleConstraint, or a DihedralConstraint, pull out the pair of residues
	/// that are constrained, and return the pair.
	/// @details Throws if type is unrecognized or if more than two residues are
	/// constrained.  Values of res1 and res2 are overwritten by this operation.
	void get_residues_from_constraint(
		Size & res1,
		Size & res2,
		core::scoring::constraints::ConstraintCOP const & cst
	) const;

	/// @brief Given a constraint, determine if it is an AtomPairConstraint, pull
	/// out the pair of residues that are constrained, and return the pair.
	/// @details If successful, values of res1 and res2 are overwritten by this
	/// operation, and the function returns "true".  Otherwise, values are not
	/// altered, and the function returns "false".
	bool get_residues_from_atom_pair_constraint(
		Size & res1,
		Size & res2,
		core::scoring::constraints::ConstraintCOP const & cst
	) const;

	/// @brief Do all-atom FastRelax refinement.
	void do_fullatom_refinement(
		core::pose::Pose & pose,
		utility::vector1< bool > move_bb
	) const;

};

////////////////////////////////////////////////////////////////////////////////

//////////////////////////
/// @note Constructors ///
//////////////////////////

////////////////////////////////////////////////////////////////////////////////
/// @brief Construct a Default Position Mover
DomainAssembly::DomainAssembly() :
	Mover(), //初始化基类
	full_seq_(), //存储完成序列信息
	infiles_(), //文件
	poses_(), //用于存储蛋白质结构信息
	sequences_(), //序列
	offsets_(),
	linkers_(),
	movemap_()
{}

////////////////////////////////////////////////////////////////////////////////
/// @brief Copy Constructor
/// @details Make a deep copy of this mover object
//构造函数定义，成员初始化列表，对类成员进行初始化
DomainAssembly::DomainAssembly( DomainAssembly const & src ) :
	Mover( src ),
	full_seq_( src.full_seq_ ),
	tmpdb_( src.tmpdb_ ),
	infiles_( src.infiles_ ),
	poses_( src.poses_ ),
	sequences_( src.sequences_ ),
	offsets_( src.offsets_ ),
	linkers_( src.linkers_ ),
	movemap_( src.movemap_ )
{}

////////////////////////////////////////////////////////////////////////////////
/// @brief Assignment Operator
/// @details Make a deep copy of this mover object, overriding the assignment operator
DomainAssembly &
DomainAssembly::operator=( DomainAssembly const & src )
{
	// Abort self-assignment.
	if ( this == &src ) {
		return *this;
	}

	// Otherwise, create a new object
	return *( new DomainAssembly( *this ) );
}

////////////////////////////////////////////////////////////////////////////////
/// @brief Destructor
DomainAssembly::~DomainAssembly() = default;

///////////////////////////
/// @note Mover Methods ///
///////////////////////////

////////////////////////////////////////////////////////////////////////////////
/// @brief Get the name of this mover
std::string
DomainAssembly::get_name() const {
	return "DomainAssembly";
}

////////////////////////////////////////////////////////////////////////////////
/// @brief run domain assembly
// 剪切点变种（Cutpoint Variants）是蛋白质中的一个特殊构象，通常用于表示多肽链的连接部分。在蛋白质多肽链中，
//蛋白质的不同片段之间通常通过共价键连接在一起，这种连接通常是蛋白质的主链骨架之间的键。这些连接点就是剪切点
void DomainAssembly::apply( Pose & pose ) {

	TR << "Calling DomainAssembly" << std::endl;

	register_options();
	init_from_cmd();

	// clear vector contents, needed for running things in parallel
	pose.clear();
	poses_.clear();
	sequences_.clear();
	offsets_.clear();


	////////////////////////////////////////////////////////////////////////////////////////
	/// @note SETUP

	// read in poses_ 文件中读取蛋白质信息
	utility::vector1< core::pose::Pose > poses_ = core::import_pose::poses_from_files( infiles_ );
	chains_ = poses_;
//	Pose tmdomain = poses_[ tmpdb_ ];


    // get length of the full-length protein 获取全长蛋白质
	nres = full_seq_.size();
	


	// get sequences_ from poses_
	for ( Size i = 1; i <= poses_.size(); ++i ) {
		sequences_.push_back( poses_[i].sequence() );
	}
	Size chain1_length = sequences_[1].size();
	TR << "POSES " << poses_.size() << std::endl;
	TR << "SEQUENCES " << sequences_.size() << std::endl;
	TR << "SEQUENCES 1 : " << sequences_[1].size() << std::endl << sequences_[1] << std::endl;
	TR << "SEQUENCES 2 : " << sequences_[2].size() << std::endl << sequences_[2] << std::endl;


    
	// read CDR_index.txt
	ifstream index_path( index_path_.c_str() );
	if ( !index_path ) {
		TR << "Please provide cdr index file!" << std::endl;
		exit(0);
	}
	// load CDR_index
    TR << "loading cdr index..." << std::endl;
    std::string line;
    while (std::getline(index_path, line)) {

        std::istringstream iss(line);
        Size start, end;
        if (!(iss >> start >> end)) {
            TR << "Error reading index file format!" << std::endl;
            exit(0);
        }
        cdr_starts.push_back(start);
        cdr_ends.push_back(end);

    }

    index_path.close();
	TR << "cdr_starts: ";
	for (const auto& start : cdr_starts) {
		TR << start << " ";
	}
	TR << std::endl;
	TR << "cdr_ends: ";
	for (const auto& end : cdr_ends) {
		TR << end << " ";
	}
	TR << std::endl;


	// 读取文件中的内容 注意cdr_starts只能读取第一列的数 cdr_ends读取第二列的数
	// 假设txt中内容为	 25 32
					//	56 72
					//	102 123

	nres_ab = sequences_[1].size();


	for (Size i = 1; i <= cdr_starts.size(); ++i) {
        movable_residues_.push_back(make_pair(cdr_starts[i], cdr_ends[i]));
    }

    // 将不可移动的残基对存储在immovable_residues_中 √
    // 分别处理每段不可移动的区域
    Size previous_end = 0;
    for (Size i = 1; i <= movable_residues_.size(); ++i) {
//        int current_start = cdr_starts[i];
        Size current_start = movable_residues_[i].first;
        if (previous_end + 1 < current_start) {
            immovable_residues_.push_back(make_pair(previous_end + 1, current_start - 1));
        }
//        previous_end = cdr_ends[i];
        previous_end = movable_residues_[i].second;
    }
    // 处理最后一段不可移动的区域
    if (previous_end < nres_ab) {
        immovable_residues_.push_back(make_pair(previous_end + 1, nres_ab));
    }


	TR << "movable residues: " << std::endl;
	for ( Size i = 1; i <= movable_residues_.size(); ++i ) {
		TR << i << " " << movable_residues_[i] << std::endl;
	}

	TR << "immovable residues: " << std::endl;
	for ( Size i = 1; i <= immovable_residues_.size(); ++i ) {
		TR << i << " " << immovable_residues_[i] << std::endl;
	}

	////////////////////////////////////////////////////////////////////////////////////////
	/// @note CREATE MOVEMAP

	// create a MoveMap and set the backbone moveable to the linkers_ plus 1 flanking residue on either side 设定链接处和侧链是可移动的
	TR << "finding movable residues" << std::endl;
	utility::vector1< bool > move_bb( sequences_[1].size()+sequences_[2].size()+1, false );
	for ( Size i = 1; i <= movable_residues_.size(); ++i ) {
		for ( Size j = movable_residues_[i].first; j <= movable_residues_[i].second; ++j )
			move_bb[j] = true;
	}

	// finally create the MoveMap（标记了那些残基可以移动）
	TR << "creating a MoveMap" << std::endl;
	core::kinematics::MoveMapOP movemap_( new core::kinematics::MoveMap() );
	movemap_->set_bb( move_bb );
// 	movemap_->show();
//	/ @note READ CONSTRAINTS antiberty (antibody-antigen inter-distance)
//
//	 read distance constraints3
	ifstream dist_path1( dist_path1_.c_str() );
	if ( !dist_path1 ) {
		TR << "Please provide distance constraint file!" << std::endl;
		exit(0);
	}

    // load distance constraints
    TR << "loading distance constraints1..." << std::endl;

	std::string dist_line1;
	utility::vector1< utility::vector1< Real > > dist_constr1;
	Size line_number1 = 1;
	Size chain_number1 = 2;

	while ( getline( dist_path1, dist_line1 ) ) {
		istringstream data( dist_line1 );
        utility::vector1< Real > line_dist_constr1;
        Real distance_value;

		for ( Size m = 1; m <= sequences_[chain_number1].size(); ++m ) {
            data >> distance_value;
            line_dist_constr1.push_back( distance_value );
        }
        dist_constr1.push_back( line_dist_constr1 );
        if ( line_number1 == sequences_[1].size() ) {
            line_number1 = 1;
            chain_number1++;
            dist_constr1_.push_back( dist_constr1 );
            dist_constr1.clear();
		}
		else
            line_number1++;
	}

	dist_path1.close();

// ////////////////////////////////////////////////////////////////////////////////////////
// 	/// @note READ CONSTRAINTS AFM (antibody-antigen inter-distance)

// 	// read distance constraints4
	ifstream dist_path2( dist_path2_.c_str() );
	if ( !dist_path2 ) {
		TR << "Please provide distance constraint file!" << std::endl;
		exit(0);
	}

    // load distance constraints
    TR << "loading distance constraints2..." << std::endl;

	std::string dist_line2;
	utility::vector1< utility::vector1< Real > > dist_constr2;
	Size line_number2 = 1;
	Size chain_number2 = 2;

	while ( getline( dist_path2, dist_line2 ) ) {
		istringstream data( dist_line2 );
        utility::vector1< Real > line_dist_constr2;
        Real distance_value;

		for ( Size m = 1; m <= sequences_[chain_number2].size(); ++m ) {
            data >> distance_value;
            line_dist_constr2.push_back( distance_value );
        }
        dist_constr2.push_back( line_dist_constr2 );

        if ( line_number2 == sequences_[1].size() ) {
            line_number2 = 1;
            chain_number2++;
            dist_constr2_.push_back( dist_constr2 );
            dist_constr2.clear();
		}
		else
            line_number2++;
	}
	dist_path2.close();

	////////////////////////////////////////////////////////////////////////////////////////
	/// @note INITIALIZE POPULATION

    // 生成一个具有不同构象的蛋白质构象集合  初始化种群 5个

	TR << "population initialization..." << std::endl;

	for ( Size n = 1; n <= 20; ++n ) { //4
	    TR << "************* Big agitation *************"<< std::endl;
        Pose best_pos;
        Real best_FAPE = 100.0;
        core::Real kT = 2.0;  // 设置温度
        core::Size n_moves = 5;  // 设置一次move的扰动次数

        protocols::simple_moves::SmallMover smallmover(movemap_, kT, n_moves);
        smallmover.angle_max( 'H', 0.0 );
        smallmover.angle_max( 'E', 0.0 );
        smallmover.angle_max( 'L', 5.0 );
        
		Pose init_pose( poses_[1] );  //1个
		smallmover.apply(init_pose);
//		//每个构象生成中复制已组装好的完整蛋白质结构 full_pose，以 init_pose 作为当前构象的初始蛋白质结构。
//	    //进行relax操作

		population_pose_.push_back( init_pose );  //将生成的构象 init_pose 添加到构象集合 population_pose_ 中 20个

        // 计算该构象的得分，这里使用了两个评分指标：
		Real init_total_score( 0.5*frame_aligned_point_error_CA_CDR_v1( init_pose, poses_ ) + 0.5*frame_aligned_point_error_CA_CDR_v2( init_pose, poses_ ) );
		TR << "Init Score_" << init_total_score << std::endl;
		population_score_.push_back( init_total_score );

        TR << "************* Small agitation *************"<< std::endl;
        for ( Size m = 1; m <= 100; ++m ) {

            Pose mutation_pos;
            mutation_pos = population_pose_[n];
            core::Real kT2 = 2.0;  // 设置温度
            core::Size n_moves2 = 1;  // 设置移动次数
            protocols::simple_moves::SmallMover smallmover2(movemap_, kT2, n_moves2);
            smallmover2.angle_max( 'H', 0.0 );
            smallmover2.angle_max( 'E', 0.0 );
            smallmover2.angle_max( 'L', 2.0 );
            smallmover2.apply(mutation_pos);

            Real trail_score( 0.5*frame_aligned_point_error_CA_CDR_v1( mutation_pos, poses_ ) + 0.5*frame_aligned_point_error_CA_CDR_v2( mutation_pos, poses_ ) );
            Real target_score(population_score_[n]);

            bool success1( greedy_accept( target_score, trail_score ) );
            if ( success1 ) {
                population_pose_[n] = mutation_pos;  //角度更新
                population_score_[n] = trail_score;  //能量分数更新
            }

            if ( population_score_[n] < best_FAPE ) {
                best_pos = population_pose_[n];
                best_FAPE = population_score_[n];  //一共有100个
                TR << "[ " << m << " / " << 10000 << " ]...    FAPE: " << best_FAPE << std::endl;
            }
        }

        Real Final_score( 0.5*frame_aligned_point_error_CA_CDR_v1( population_pose_[n], poses_ ) + 0.5*frame_aligned_point_error_CA_CDR_v2( population_pose_[n], poses_ ) );

        stringstream pose_name;
		processinformation << "save pose_" << n << "_" << Final_score << std::endl;
		pose_name << "./ensemble_pdb/pose_" << n << "_" << Final_score << "_"<< ".pdb";
		population_pose_[n].dump_pdb(pose_name.str());
		pose_name.str("");
		pose_name.clear();

		stringstream init_pose_name;
		init_pose_name << "./ensemble_pdb/init_pose_"<< init_total_score <<".pdb";
		poses_[1].dump_pdb(init_pose_name.str());
		init_pose_name.str("");
		init_pose_name.clear();

	}



} // apply

////////////////////////////////////////////////////////////////////////////////
/// @brief Register Options with JD2
void DomainAssembly::register_options() {

	using namespace basic::options;
	option.add_relevant( OptionKeys::in::file::fasta );
	// option.add_relevant( OptionKeys::in::file::fasta1 );
	option.add_relevant( OptionKeys::mp::assembly::poses );
//	option.add_relevant( OptionKeys::mp::assembly::TM_pose_number );
	option.add_relevant( OptionKeys::relax::range::angle_max );
//	option.add_relevant( OptionKeys::in::file::native );

} // register options

////////////////////////////////////////////////////////////////////////////////
/// @brief Initialize Mover options from the comandline
void DomainAssembly::init_from_cmd() {

	using namespace basic::options;

	// cry if PDB list not given  通过检查命令行参数来读取和解析程序所需的配置信息。
	// 如果用户在命令行中提供了 -in:file:fasta 选项，那么它会读取FASTA文件，并将其中的序列存储在 full_seq_ 变量中。
	if ( option[OptionKeys::in::file::fasta].user() ) {
		full_seq_ = core::sequence::read_fasta_file( option[ OptionKeys::in::file::fasta ]()[1] )[1]->sequence();
		TR << "Read in fasta file " << option[OptionKeys::in::file::fasta]()[1] << std::endl;
	} else {
		throw CREATE_EXCEPTION(utility::excn::Exception, "Please provide fasta file with -in:file:fasta!");
	}


	// read in PDB list 如果用户在命令行中提供了 -mp:assembly:poses 选项，它会读取指定路径下的一系列PDB文件，
	// 并将它们的文件路径存储在 infiles_ 变量中。这些PDB文件通常用于构建装配模型。
	if ( option[mp::assembly::poses].user() ) {
		utility::vector1< std::string > domain_path_ = basic::options::option[mp::assembly::poses]();
		std::string domain_path = domain_path_[1];

		for ( Size n = 1; n < 100 ; ++n ) {
		    stringstream in_pose;
		    in_pose << domain_path << "/dom" << n << ".pdb";
		    string pdb_route;
		    in_pose >> pdb_route;

		    ifstream pdb_route_( pdb_route.c_str() );
	        if ( pdb_route_ )
		        infiles_.push_back( pdb_route );
		}
		if ( infiles_.size() == 0 )
		    throw CREATE_EXCEPTION(utility::excn::Exception, "Please provide domain PDB files!");

	} else {
		throw CREATE_EXCEPTION(utility::excn::Exception, "Please provide route of domain PDB files!");
	}

    tmpdb_ = infiles_.size();   //域的数量
	cout << "tmpdb_: " << tmpdb_ << std::endl;


} // init from commandline

////////////////////////////////////////////////////////////////////////////////
/// @brief create residue
Residue DomainAssembly::create_residue_from_resn( Pose & pose, Size resnumber ) {
// 这个函数的主要目的是创建一个 Residue 对象，用于表示在给定的Pose（蛋白质结构）中的特定残基
	using namespace core::conformation;

	std::string name3 = pose.residue( resnumber ).name3();

	ResidueTypeSetCOP residue_set( ChemicalManager::get_instance()->residue_type_set( core::chemical::FA_STANDARD ) );
	ResidueType const & rsd_type( residue_set->name_map( name3 ) );

	// true: preserve Cbeta
	ResidueOP rsd( ResidueFactory::create_residue( rsd_type, pose.residue( resnumber ), pose.conformation(), true ) );

	Residue rsd1( *rsd );

	return rsd1;

} // create residue from tlc

////////////////////////////////////////////////////////////////////////////////
/// @brief create residue
//  根据给定的单字母代码 (OLC: One-Letter Code) 创建一个 Residue 对象
ResidueOP DomainAssembly::create_residue_from_olc( char olc ) {

	using namespace core::chemical;
	ResidueTypeSetCOP const & residue_set( ChemicalManager::get_instance()->residue_type_set( core::chemical::FA_STANDARD ) );

	ResidueTypeCOP rsd_type( residue_set->get_representative_type_name1( olc ) );
	ResidueType const & res( *rsd_type );
	ResidueOP rsd( ResidueFactory::create_residue( res ) );

	return rsd;
} // create residue from OLC

////////////////////////////////////////////////////////////////////////////////
/// @brief set random torsion
void DomainAssembly::set_random_torsion( Pose & pose, Size resn ) {

	// get random phi and psi
	core::Real random_phi( numeric::random::uniform() * 360 - 180 );
	core::Real random_psi( numeric::random::uniform() * 360 - 180 );

	// set phi and psi to random and omega to 180
	pose.set_phi( resn, random_phi );
	pose.set_psi( resn, random_psi );
	pose.set_omega( resn, 180 );

} // set random torsion

/// @brief set random rotation and translation
utility::vector1< utility::vector1< Real > > DomainAssembly::set_random_rot_tra() {

    utility::vector1< utility::vector1< Real > > random_rot_tra_;
    for ( Size n = 1; n < tmpdb_; ++n ) {

        utility::vector1< Real > random_rot_tra;

    	// get random euler angles
	    core::Real random_euler_x( numeric::random::uniform() * 360 - 180 );
	    core::Real random_euler_y( numeric::random::uniform() * 180 - 90 );
	    core::Real random_euler_z( numeric::random::uniform() * 360 - 180 );

	    // get random spherical radius and angles
	    core::Real random_spherical_radius( numeric::random::uniform() * 200 - 100 );
	    core::Real random_spherical_theta( numeric::random::uniform() * 200 - 100 );
	    core::Real random_spherical_phi( numeric::random::uniform() * 200 - 100 );

	    random_rot_tra.push_back( random_euler_x );
	    random_rot_tra.push_back( random_euler_y );
	    random_rot_tra.push_back( random_euler_z );
	    random_rot_tra.push_back( random_spherical_radius );
	    random_rot_tra.push_back( random_spherical_theta );
	    random_rot_tra.push_back( random_spherical_phi );


	    random_rot_tra_.push_back( random_rot_tra );
    }

    return random_rot_tra_;
}

////////////////////////////////////////////////////////////////////////////////
/// @brief Print score to cout
void DomainAssembly::print_score( Pose & pose, core::scoring::ScoreFunctionOP sfxn ) {

	// print energies and iteration
	Real tot_score = ( *sfxn )( pose );
	Real fa_rep = pose.energies().total_energies()[ scoring::fa_rep ];
	Real fa_atr = pose.energies().total_energies()[ scoring::fa_atr ];

	pose.energies().show_total_headers( TR );
	TR << std::endl;
	pose.energies().show_totals( TR );
	TR << std::endl;

	TR << "tot: " << tot_score << " fa_rep: " << fa_rep << " fa_atr: " << fa_atr << std::endl;

} // print score

////////////////////////////////////////////////////////////////////////////////
/// @brief calculate RMSD
Real DomainAssembly::cal_rmsd( Pose & pose ) {
	Real rmsd;
//	rmsd = core::scoring::CA_rmsd( native, pose );
    rmsd = -1.0;
	return rmsd;
}

////////////////////////////////////////////////////////////////////////////////
/// @brief calculate total score
Real DomainAssembly::cal_total_score( Pose & pose ) {
	Real total_score;
	total_score = ( *sfxn_ )( pose );
	return total_score;
}

////////////////////////////////////////////////////////////////////////////////
/// @brief calculate ref total score
Real DomainAssembly::cal_ref_total_score( Pose & pose ) {
	Real ref_total_score;
	ref_total_score = ( *sfxn_fullatom_ )( pose );
	return ref_total_score;
}

////////////////////////////////////////////////////////////////////////////////
/// @brief defines reference frame per residue using backbone atoms
utility::vector1< utility::vector1< Real > > DomainAssembly::set_lframe( Pose & pose, Size res ) {
// 其主要作用是计算给定氨基酸残基的局部坐标系（Local Frame）。
// 这个局部坐标系由三个互相垂直的单位矢量组成，通常用于描述蛋白质中残基的取向
    utility::vector1< utility::vector1< Real > > lframe;

    numeric::xyzVector< Real > N = pose.residue(res).xyz("N");
    numeric::xyzVector< Real > CA = pose.residue(res).xyz("CA");
    numeric::xyzVector< Real > C = pose.residue(res).xyz("C");

    utility::vector1< Real > v1;
    utility::vector1< Real > v2;
    utility::vector1< Real > e1;
    utility::vector1< Real > u2;
    utility::vector1< Real > e2;
    utility::vector1< Real > e3;

    for ( Size i = 0; i < 3; ++i ) {
        v1.push_back( C[i] - CA[i] );
        v2.push_back( N[i] - CA[i] );
    }

    Real v1_norm = sqrt( v1[1] * v1[1] + v1[2] * v1[2] + v1[3] * v1[3] );

    for ( Size i = 1; i <= 3; ++i )
        e1.push_back( v1[i] / v1_norm );

    Real e1_mul_v2 = e1[1] * v2[1] + e1[2] * v2[2] + e1[3] * v2[3];

    for ( Size i = 1; i <= 3; ++i )
        u2.push_back( v2[i] - e1[i] * e1_mul_v2 );

    Real u2_norm = sqrt( u2[1] * u2[1] + u2[2] * u2[2] + u2[3] * u2[3] );

    for ( Size i = 1; i <= 3; ++i )
        e2.push_back( u2[i] / u2_norm );

    e3.push_back( e1[2] * e2[3] - e1[3] * e2[2] );
    e3.push_back( e1[3] * e2[1] - e1[1] * e2[3] );
    e3.push_back( e1[1] * e2[2] - e1[2] * e2[1] );

    lframe.push_back( e1 );
    lframe.push_back( e2 );
    lframe.push_back( e3 );

    return lframe;
}


////////////////////////////////////////////////////////////////////////////////
// @brief calculate distance score
Real DomainAssembly::cal_Cscore( Pose & pose , utility::vector1< core::pose::Pose > poses_) {

    Real clash_score = 0;
    utility::vector1< utility::vector1< Real > > mat1;
    for (Size i = 1; i <= movable_residues_.size(); ++i){

        for ( Size res1 = movable_residues_[i].first; res1 <= movable_residues_[i].second; ++res1 ) {

            for ( Size res2 = 1; res2 <= sequences_[2].size(); ++res2 ) {

                mat1 = set_lframe( pose, res1 );

                numeric::xyzVector< Real > CA1 = pose.residue(res1).xyz("CA");
                utility::vector1< Real > t1;
                t1.push_back( CA1[0] );
                t1.push_back( CA1[1] );
                t1.push_back( CA1[2] );

                numeric::xyzVector< Real > CA2 = poses_[2].residue(res2).xyz("CA");
                utility::vector1< Real > x_global;
                x_global.push_back( CA2[0] );
                x_global.push_back( CA2[1] );
                x_global.push_back( CA2[2] );

                utility::vector1< Real > x;
                for ( Size i = 1; i <= 3; ++i )
                    x.push_back( mat1[i][1] * (x_global[1]-t1[1]) + mat1[i][2] * (x_global[2]-t1[2]) + mat1[i][3] * (x_global[3]-t1[3]) );

                Real dist = sqrt(x[1]* x[1] + x[2]* x[2]+ x[3]* x[3]);

                Real clash_penalty = 0;
                if (dist < 3.75) {
                    clash_penalty = 1 / dist;
                }
                else {
                    clash_penalty = 0;
                }
                clash_score += clash_penalty;
            }
        }
    }

	return clash_score;
}

////////////////////////////////////////////////////////////////////////////////
// @brief calculate distance score
Real DomainAssembly::cal_Cscore_antibody( Pose & pose ) {

    Real clash_score = 0;
    utility::vector1< utility::vector1< Real > > mat1;
//    for (Size i = 1; i <= movable_residues_.size(); ++i){

//    for ( Size res1 = movable_residues_[i].first; res1 <= movable_residues_[i].second; ++res1 ) {
    for ( Size res1 = 1; res1 <= sequences_[1].size(); ++res1 ) {

        for ( Size res2 = 1; res2 <= sequences_[1].size(); ++res2 ) {

            mat1 = set_lframe( pose, res1 );

            numeric::xyzVector< Real > CA1 = pose.residue(res1).xyz("CA");
            utility::vector1< Real > t1;
            t1.push_back( CA1[0] );
            t1.push_back( CA1[1] );
            t1.push_back( CA1[2] );

            numeric::xyzVector< Real > CA2 = pose.residue(res2).xyz("CA");
            utility::vector1< Real > x_global;
            x_global.push_back( CA2[0] );
            x_global.push_back( CA2[1] );
            x_global.push_back( CA2[2] );

            utility::vector1< Real > x;
            if ( res1 != res2){
                for ( Size i = 1; i <= 3; ++i )
                    x.push_back( mat1[i][1] * (x_global[1]-t1[1]) + mat1[i][2] * (x_global[2]-t1[2]) + mat1[i][3] * (x_global[3]-t1[3]) );

                Real dist = sqrt(x[1]* x[1] + x[2]* x[2]+ x[3]* x[3]);

                Real clash_penalty = 0;
                if (dist < 3.75) {
                    clash_penalty = 1 / dist;
                }
                else {
                    clash_penalty = 0;
                }
                clash_score += clash_penalty;
            }
        }
    }
	return clash_score;
}



/// @brief calculate frame aligned point error (CA)  预测结构，输入的原始结构
Real DomainAssembly::frame_aligned_point_error_CA_CDR_v1 (Pose & pose , utility::vector1< core::pose::Pose > poses_) {
   
    Real FAPE = 0;
	Real total_dist_diff = 0;
    Real count = 0;
    utility::vector1< utility::vector1< Real > > mat1;
    for (Size i = 1; i <= movable_residues_.size(); ++i){
		
        for ( Size res1 = movable_residues_[i].first; res1 <= movable_residues_[i].second; ++res1 ) {
			
            for ( Size res2 = 1; res2 <= sequences_[2].size(); ++res2 ) {
				
                if ( dist_constr1_[1][res1][res2] > 0) {

                    mat1 = set_lframe( pose, res1 ); //以（抗体）预测结构中𝑟1残基为中心构建的局部旋转矩阵；
                    numeric::xyzVector< Real > CA1 = pose.residue(res1).xyz("CA");
                    utility::vector1< Real > t1;
                    t1.push_back( CA1[0] );
                    t1.push_back( CA1[1] );
                    t1.push_back( CA1[2] );
					
                    numeric::xyzVector< Real > CA2 = poses_[2].residue(res2).xyz("CA");//将抗原结构中𝑟2残基的坐标投影到以𝑟1为中心的局部坐标系中；
                    utility::vector1< Real > x_global;
                    x_global.push_back( CA2[0] );
                    x_global.push_back( CA2[1] );
                    x_global.push_back( CA2[2] );
					
                    utility::vector1< Real > x;

                    for ( Size i = 1; i <= 3; ++i )
                        x.push_back( mat1[i][1] * (x_global[1]-t1[1]) + mat1[i][2] * (x_global[2]-t1[2]) + mat1[i][3] * (x_global[3]-t1[3]) );
					
                    Real dist = sqrt(x[1]* x[1] + x[2]* x[2]+ x[3]* x[3]);
                    Real pdist = dist_constr1_[1][res1][res2];
                    FAPE += sqrt((dist - pdist) * (dist - pdist));
                    count++;
                }
            }
        }
    }
    FAPE /= count;
    return FAPE;
}

Real DomainAssembly::frame_aligned_point_error_CA_CDR_v2( Pose & pose , utility::vector1< core::pose::Pose > poses_) {

    Real FAPE = 0;
	Real total_dist_diff = 0;
    Real count = 0;
    utility::vector1< utility::vector1< Real > > mat2;
    for (Size i = 1; i <= movable_residues_.size(); ++i){

        for ( Size res1 = movable_residues_[i].first; res1 <= movable_residues_[i].second; ++res1 ) {

            for ( Size res2 = 1; res2 <= sequences_[2].size(); ++res2 ) {

                if ( dist_constr2_[1][res1][res2] > 0) {

                    mat2 = set_lframe( pose, res1 );

                    numeric::xyzVector< Real > CA1 = pose.residue(res1).xyz("CA");
                    utility::vector1< Real > t1;
                    t1.push_back( CA1[0] );
                    t1.push_back( CA1[1] );
                    t1.push_back( CA1[2] );

                    numeric::xyzVector< Real > CA2 = poses_[2].residue(res2).xyz("CA");
                    utility::vector1< Real > x_global;
                    x_global.push_back( CA2[0] );
                    x_global.push_back( CA2[1] );
                    x_global.push_back( CA2[2] );

                    utility::vector1< Real > x;

                    for ( Size i = 1; i <= 3; ++i )
                        x.push_back( mat2[i][1] * (x_global[1]-t1[1]) + mat2[i][2] * (x_global[2]-t1[2]) + mat2[i][3] * (x_global[3]-t1[3]) );

                    Real dist = sqrt(x[1]* x[1] + x[2]* x[2]+ x[3]* x[3]);

                    Real pdist = dist_constr2_[1][res1][res2];

                    FAPE += sqrt((dist - pdist) * (dist - pdist));
                    count++;
                }
            }
        }
    }
    FAPE /= count;

    return FAPE;
}

Real DomainAssembly::score_rot_tra_v3( utility::vector1< utility::vector1< Real > > rot_tra_ ) {

    Real total_dist_diff3 = 0;
    Real count3 = 0;
//    for ( Size i = 1; i < tmpdb_; ++i ) {
    for ( Size i = 1; i < 2; ++i ) {
//        for ( Size j = i + 1; j <= tmpdb_; ++j ) {
        for ( Size j = 2; j <= tmpdb_; ++j ) {
            utility::vector1< Real > rot_tra_1, rot_tra_2;
            if ( i == 1 )
                rot_tra_1.resize(6, 0.0);
            else
                rot_tra_1 = rot_tra_[i - 1];

            rot_tra_2 = rot_tra_[j - 1];

            utility::vector1< utility::vector1< Real > > rotMat1( EulerAngles_to_rotationMatrix( rot_tra_1[1], rot_tra_1[2], rot_tra_1[3] ) );
            utility::vector1< utility::vector1< Real > > rotMat2( EulerAngles_to_rotationMatrix( rot_tra_2[1], rot_tra_2[2], rot_tra_2[3] ) );
            utility::vector1< Real > traVec1{ rot_tra_1[4], rot_tra_1[5], rot_tra_1[6] };
            utility::vector1< Real > traVec2{ rot_tra_2[4], rot_tra_2[5], rot_tra_2[6] };

            for ( Size res1 = 1; res1 <= sequences_[i].size(); ++res1 ) {
                for ( Size res2 = 1; res2 <= sequences_[j].size(); ++res2 ) {
                    if ( dist_constr3_[j-1][res1][res2] > 0) {
                        utility::vector1< Real > CA_coord_after_trans_1, CA_coord_after_trans_2;
                        numeric::xyzVector< Real > CA_coord_1 = chains_[i].residue(res1).xyz("CA");
                        numeric::xyzVector< Real > CA_coord_2 = chains_[j].residue(res2).xyz("CA");

                        for ( Size k = 1; k <= 3; ++k ) {
                            CA_coord_after_trans_1.push_back( rotMat1[k][1] * (CA_coord_1[0] - traVec1[1]) + \
                                                              rotMat1[k][2] * (CA_coord_1[1] - traVec1[2]) + \
                                                              rotMat1[k][3] * (CA_coord_1[2] - traVec1[3]) );
                            CA_coord_after_trans_2.push_back( rotMat2[k][1] * (CA_coord_2[0] - traVec2[1]) + \
                                                              rotMat2[k][2] * (CA_coord_2[1] - traVec2[2]) + \
                                                              rotMat2[k][3] * (CA_coord_2[2] - traVec2[3]) );
                        }

                        Real dist = sqrt((CA_coord_after_trans_1[1] - CA_coord_after_trans_2[1]) * (CA_coord_after_trans_1[1] - CA_coord_after_trans_2[1]) + \
                                         (CA_coord_after_trans_1[2] - CA_coord_after_trans_2[2]) * (CA_coord_after_trans_1[2] - CA_coord_after_trans_2[2]) + \
                                         (CA_coord_after_trans_1[3] - CA_coord_after_trans_2[3]) * (CA_coord_after_trans_1[3] - CA_coord_after_trans_2[3]));

                        Real pdist = dist_constr3_[j-1][res1][res2];

                        total_dist_diff3 += sqrt((dist - pdist) * (dist - pdist));
                        count3++;
                    }
                }
            }
        }
    }
    if ( count3 == 0 )
        total_dist_diff3 = 1000000.0;
    else
        total_dist_diff3 /= count3;

    return total_dist_diff3;
}

Real DomainAssembly::score_rot_tra_v4( utility::vector1< utility::vector1< Real > > rot_tra_ ) {

    Real total_dist_diff4 = 0;
    Real count = 0;

//    for ( Size i = 1; i < tmpdb_; ++i ) {
    for ( Size i = 1; i < 2; ++i ) {

//        for ( Size j = i + 1; j <= tmpdb_; ++j ) {
        for ( Size j = 2; j <= tmpdb_; ++j ) {

            utility::vector1< Real > rot_tra_1, rot_tra_2;
            if ( i == 1 )
                rot_tra_1.resize(6, 0.0);
            else
                rot_tra_1 = rot_tra_[i - 1];

            rot_tra_2 = rot_tra_[j - 1];

            utility::vector1< utility::vector1< Real > > rotMat1( EulerAngles_to_rotationMatrix( rot_tra_1[1], rot_tra_1[2], rot_tra_1[3] ) );
            utility::vector1< utility::vector1< Real > > rotMat2( EulerAngles_to_rotationMatrix( rot_tra_2[1], rot_tra_2[2], rot_tra_2[3] ) );
            utility::vector1< Real > traVec1{ rot_tra_1[4], rot_tra_1[5], rot_tra_1[6] };
            utility::vector1< Real > traVec2{ rot_tra_2[4], rot_tra_2[5], rot_tra_2[6] };

            for ( Size res1 = 1; res1 <= sequences_[i].size(); ++res1 ) {

                for ( Size res2 = 1; res2 <= sequences_[j].size(); ++res2 ) {

                    if ( dist_constr2_[j-1][res1][res2] > 0) {

                        utility::vector1< Real > CA_coord_after_trans_1, CA_coord_after_trans_2;
                        numeric::xyzVector< Real > CA_coord_1 = chains_[i].residue(res1).xyz("CA");
                        numeric::xyzVector< Real > CA_coord_2 = chains_[j].residue(res2).xyz("CA");

                        for ( Size k = 1; k <= 3; ++k ) {

                            CA_coord_after_trans_1.push_back( rotMat1[k][1] * (CA_coord_1[0] - traVec1[1]) + \
                                                              rotMat1[k][2] * (CA_coord_1[1] - traVec1[2]) + \
                                                              rotMat1[k][3] * (CA_coord_1[2] - traVec1[3]) );
                            CA_coord_after_trans_2.push_back( rotMat2[k][1] * (CA_coord_2[0] - traVec2[1]) + \
                                                              rotMat2[k][2] * (CA_coord_2[1] - traVec2[2]) + \
                                                              rotMat2[k][3] * (CA_coord_2[2] - traVec2[3]) );
                        }

                        Real dist = sqrt((CA_coord_after_trans_1[1] - CA_coord_after_trans_2[1]) * (CA_coord_after_trans_1[1] - CA_coord_after_trans_2[1]) + \
                                         (CA_coord_after_trans_1[2] - CA_coord_after_trans_2[2]) * (CA_coord_after_trans_1[2] - CA_coord_after_trans_2[2]) + \
                                         (CA_coord_after_trans_1[3] - CA_coord_after_trans_2[3]) * (CA_coord_after_trans_1[3] - CA_coord_after_trans_2[3]));

                        Real pdist = dist_constr2_[j-1][res1][res2];

                        total_dist_diff4 += sqrt((dist - pdist) * (dist - pdist));
                        count++;
                    }
                }
            }
        }
    }

    if ( count == 0 )
        total_dist_diff4 = 1000000.0;
    else
        total_dist_diff4 /= count;

    return total_dist_diff4;
}

utility::vector1< utility::vector1< Real > > DomainAssembly::matrix_multiply( utility::vector1< utility::vector1< Real > > arrA, utility::vector1< utility::vector1< Real > > arrB ) {

	//矩阵arrA的行数
	Size rowA = arrA.size();
	//矩阵arrA的列数
	Size colA = arrA[1].size();
	//矩阵arrB的行数
	Size rowB = arrB.size();
	//矩阵arrB的列数
	Size colB = arrB[1].size();
	//相乘后的结果矩阵
	utility::vector1< utility::vector1< Real > > res;
	if ( colA != rowB ) {//如果矩阵arrA的列数不等于矩阵arrB的行数。则返回空
		return res;
	}
	else {
		//设置结果矩阵的大小，初始化为为0
		res.resize( rowA );
		for ( Size i = 1; i <= rowA; ++i ) {
			res[i].resize( colB );
		}

		//矩阵相乘
		for ( Size i = 1; i <= rowA; ++i ) {
			for ( Size j = 1; j <= colB; ++j ) {
				for ( Size k = 1; k <= colA; ++k ) {
					res[i][j] += arrA[i][k] * arrB[k][j];
				}
			}
		}
	}
	return res;
}

utility::vector1< utility::vector1< Real > > DomainAssembly::EulerAngles_to_rotationMatrix( Real euler_x, Real euler_y, Real euler_z ) {

    euler_x = euler_x * 3.141592653589793 / 180.0;
    euler_y = euler_y * 3.141592653589793 / 180.0;
    euler_z = euler_z * 3.141592653589793 / 180.0;

    utility::vector1< utility::vector1< Real > > R_x = {{1,            0,             0},\
                                                        {0, cos(euler_x), -sin(euler_x)},\
                                                        {0, sin(euler_x),  cos(euler_x)}};

    utility::vector1< utility::vector1< Real > > R_y = {{cos(euler_y),  0, sin(euler_y)},\
                                                        {0,             1,            0},\
                                                        {-sin(euler_y), 0, cos(euler_y)}};

    utility::vector1< utility::vector1< Real > > R_z = {{cos(euler_z), -sin(euler_z), 0},\
                                                        {sin(euler_z),  cos(euler_z), 0},\
                                                        {0,             0,            1}};

    utility::vector1< utility::vector1< Real > > R_y_x = matrix_multiply(R_y, R_x);
    utility::vector1< utility::vector1< Real > > R_z_y_x = matrix_multiply(R_z, R_y_x);

    return R_z_y_x;
}


utility::vector1< Real > DomainAssembly::spherical_to_translationVector( Real spher_r, Real spher_t, Real spher_p ) {

    spher_t = spher_t * 3.141592653589793 / 180.0;
    spher_p = spher_p * 3.141592653589793 / 180.0;

    Real x = spher_r * sin(spher_t) * cos(spher_p);
    Real y = spher_r * sin(spher_t) * sin(spher_p);
    Real z = spher_r * cos(spher_t);

    utility::vector1< Real > T;

    T.push_back(x);
    T.push_back(y);
    T.push_back(z);

    return T;
}


////////////////////////////////////////////////////////////////////////////////
/// @brief Transform the index of residue number (full to part)
// 其主要作用是将一个在 "frame"（f_index）中的索引转换为 "pose"（p_index）中的索引
// 这个函数通常用于从局部坐标系中的 "frame" 索引转换到整个蛋白质中的 "pose" 索引
Size DomainAssembly::index_f2p( Size f_index ) {
	Size p_index;
	Size num;
	for ( Size i = 1; i <= movable_residues_.size(); ++i ) {
		if ( f_index >= movable_residues_[i].first && f_index <= movable_residues_[i].second ) {
			num = 0;
			for ( Size j = 1; j <= i-1; ++j ) {
				num += ( movable_residues_[j].second - movable_residues_[j].first + 1 );
			}
			num += ( f_index - movable_residues_[i].first + 1 );
			p_index = 2 * num - 1;

			return p_index;
		}
	}

	for ( Size i = 1; i <= immovable_residues_.size(); ++i ) {
		if ( f_index >= immovable_residues_[i].first && f_index <= immovable_residues_[i].second ) {
			num = 0;
			for ( Size j = 1; j <= i-1; ++j ) {
				num += ( immovable_residues_[j].second - immovable_residues_[j].first + 1 );
			}
			num += ( f_index - immovable_residues_[i].first + 1 );
			p_index = num;

			return p_index;
		}
	}

	return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// @brief Transform the index of residue number (part to full)
// 将一个在 "pose"（p_index） 中的索引转换为 "frame"（f_index） 中的索引

Size DomainAssembly::index_p2f( bool is_axis, Size p_index ) {

	Size f_index;
	Size num;

	if ( is_axis ) {
		num = 0;
		for ( Size i = 1; i <= movable_residues_.size(); ++i ) {
			for ( Size j = movable_residues_[i].first; j <= movable_residues_[i].second; ++j ) {
				++num;
				if ( num == (p_index + 1) / 2 ) {
					f_index = j;
					return f_index;
				}
			}
		}
	} else {
		num = 0;
		for ( Size i = 1; i <= immovable_residues_.size(); ++i ) {
			for ( Size j = immovable_residues_[i].first; j <= immovable_residues_[i].second; ++j ) {
				++num;
				if ( num == p_index ) {
					f_index = j;
					return f_index;
				}
			}
		}
	}

	return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// @brief Accepted with Boltzmann probability
// 根据玻尔兹曼分布（Boltzmann distribution）来判断是否接受一个能量变化

bool DomainAssembly::boltzmann_accept( const Real targetEnergy, const Real trialEnergy, Real recipocal_KT ) {
	if ( trialEnergy <= targetEnergy )
		return true;
	else {
		Real probability = exp( -(trialEnergy - targetEnergy) / recipocal_KT );
		if ( probability >= numeric::random::rg().uniform() )
			return true;
		else
			return false;
	}
}

////////////////////////////////////////////////////////////////////////////////
/// @brief Score of rotation angle
// 评估在应用一组旋转角度后的构象的质量，其中分数由 frame_aligned_point_error_CA 函数给出，
// 它通常用于描述构象的质量或准确性。函数返回分数值作为评估结果，用于比较不同构象之间的相对质量。
std::pair<Pose, Real> DomainAssembly::score_rotation_angle_v1( Pose & pose, utility::vector1< Real > & rotation_angle , utility::vector1< core::pose::Pose > poses_ ) {
//    PoseScore result;
//    result.score = frame_aligned_point_error_CA(pose);
//    result.pose = pose;
    Pose updatepose = pose;
	// TR << "updatepose.size(): "<< updatepose.size() << std::endl;
	// TR << "rotation_angle.size(): "<< rotation_angle.size() << std::endl;
    for( Size j = 1; j <= 3; ++j){
        
        for ( Size r = movable_residues_[j].first; r <= movable_residues_[j].second; ++r ) {
            updatepose.set_phi(r, updatepose.phi(r) + rotation_angle[index_f2p(r)]);
            updatepose.set_psi(r, updatepose.psi(r) + rotation_angle[index_f2p(r) + 1]);
        }
        
    }
	
    Real score = frame_aligned_point_error_CA_CDR_v1(updatepose, poses_);  //能量
	
    return std::make_pair(updatepose,score);
//    return frame_aligned_point_error_CA( pose );
//    return result;
}
////////////////////////////////////////////////////////////////////////////////
/// @brief Solving the rotation angle by differential evolution algorithm
// 这段代码实现了旋转角度优化的算法，用于寻找最佳的旋转角度，以最小化蛋白质构象的 Frame Aligned Point Error (FAPE)。
std::pair<utility::vector1<Real>,utility::vector1<core::pose::Pose>> DomainAssembly::rotation_angle_optimization( Size index ,utility::vector1< core::pose::Pose > poses_) {
    Real best_FAPE = 100.0;
    utility::vector1< Real > best_angle;
    Pose pose_from_population;
	utility::vector1< utility::vector1< Real > > population_angle;
	utility::vector1< Real > population_energy;
	//********************************************************
    Size tm_index;
	utility::vector1<core::pose::Pose> mypose1;
	utility::vector1<core::pose::Pose> tmp_pose;
	utility::vector1<Size> index_vec;
	std::pair<Pose, Real> result;
	std::pair<Pose, Real> result1;
//	std::vector<Real> af2_E;
	core::pose::Pose tmpose;
	core::pose::Pose tmpose1;
	core::pose::Pose tmpose2;
	Real need_pop_size = 60;
    stringstream pose_name1;
    //寻找最低能量pose
	for ( Size n = 1; n <= NP2; ++n ) {  //NP2=6


		utility::vector1< Real > angle;
		for ( Size d = 1; d <= rotation_axis_.size(); ++d ) {
			angle.push_back( numeric::random::rg().uniform() * 2 * max_disturbance - max_disturbance );
		}

        pose_from_population = population_pose_[index];//从5个population_pose_（init_pose）中取第index索引的单个构想赋值给pose_from_population
		population_angle.push_back( angle );

        //用随机生成的angle调整pose_from_population（init_pose），并生成更新后的pose分数result1
        //result1.first中存储了更新后的构想updatepose，result1.second中存储了构象对应的分数（FAPE）
        result1 = score_rotation_angle_v1( pose_from_population, angle ,poses_);
		population_energy.push_back( result1.second);

		if ( population_energy[n] < best_FAPE ) {
            best_FAPE = population_energy[n];
//            best_pose = pose_from_population[n];
            best_angle = population_angle[n];
        }
	}

    TR << "best_FAPE size: " << population_energy.size() << std::endl;
    TR << "best_angle size: " << population_angle.size() << std::endl;

	for ( Size g = 1; g <= G; ++g ) {   // G=300
       //************************************************************
		for ( Size n = 1; n <= NP2; ++n ) {     // NP2=100
			if (g == 1) { //第一代的pose全部是原始population_pose_中得到的
                tmpose = population_pose_[index];  //单个构想
			}else{
			    if ( mypose1.size()==0){
//			        TR << "mypose1==0"<<std::endl;
                    tmpose = population_pose_[index];  //单个构想
			    }else if( mypose1.size()==1){
//                    TR << "mypose1==1"<<std::endl;
                    tmpose = mypose1[1];  //单个构想
			    }else{
				    tm_index = n % mypose1.size() + 1;  //迭代更新
				    tmpose = mypose1[tm_index];  //单个构想
				}
			}
       //************************************************************
			Size base( numeric::random::rg().random_range( 1, NP2 ) );
			Size rand1( numeric::random::rg().random_range( 1, NP2 ) );
			Size rand2( numeric::random::rg().random_range( 1, NP2 ) );

			while (rand1 == base)
				rand1 = numeric::random::rg().random_range( 1, NP2 );
			while (rand2 == base || rand2 == rand1)
				rand2 = numeric::random::rg().random_range( 1, NP2 );

			// Mutation
			// 通过两个角度以及缩放系数F来产生新的突变角度，这个新的角度是原始个体中两个角度随机现行组合，模拟了突变过程
			utility::vector1< Real > mutant_angle;

			for ( Size d = 1; d <= rotation_axis_.size(); ++d ) {
				mutant_angle.push_back( population_angle[base][d] + F * (population_angle[rand1][d] - population_angle[rand2][d]) );
			}

			// Crossover
			// 在生成的角度和原始角度之间交叉操作，交叉率控制哪些位置进行交叉
			utility::vector1< Real > cross_angle;
			Size rand_3( numeric::random::rg().random_range( 1, rotation_axis_.size() ) );
			for ( Size d = 1; d <= rotation_axis_.size(); ++d ) {
				if ( numeric::random::rg().uniform() <= CR || d == rand_3 )
					cross_angle.push_back( mutant_angle[d] );
				else
					cross_angle.push_back( population_angle[n][d] );
			}
			// Selection
			//pose_from_population = population_pose_[index];
			pose_from_population = tmpose;

			Real targetScore( population_energy[n] );
            result = score_rotation_angle_v1( pose_from_population, cross_angle, poses_ );
            Pose updata_pose(result.first);
//			tmp_pose.emplace_back(result.first);

            Real trialScore( result.second);//`result.second`表示经过旋转角度调整后的构象对象的评分（能量），是一个`Real`类型的值。
			//add 循环结束有100个pose //将变换角度后的pose加到tmp_pose中,`result.first`表示经过旋转角度调整后的构象对象，是一个`Pose`类型的对象。
			bool success( boltzmann_accept( targetScore, trialScore, 1.0 ) );
            // 如果玻尔兹曼接受，则将相应的角度以及能量值赋值给种群中
			if ( success ) {
				population_angle[n] = cross_angle;  //角度更新
				population_energy[n] = trialScore;  //能量分数更新
				tmp_pose.emplace_back(result.first);
			}

            if ( population_energy[n] < best_FAPE ) {
                best_FAPE = population_energy[n];  //一共有100个
                best_angle = population_angle[n];
            }
		}
        //************************************************************
		TR << "[ " << g << " / " << G << " ]...    FAPE: " << best_FAPE << std::endl;
		//*************************Pareto*************************************
		//从100个pose重挑选出合适的构象
		Pareto_method1(index_vec, tmp_pose, population_energy, poses_); //接受一个索引，一个变异交叉选择后的tmp_pose，一个玻尔兹曼接受后的population_energy，返回相应pose对应的索引
		// 经过这步得到index,里面存放的是筛选出来的构象的索引值
		TR << "PARETO get " << index_vec.size() << " pose" << std::endl;
//		TR << "PARETO get " << index_vec << std::endl;
		delete_unwanted_pose(tmp_pose, index_vec);
		// 将经过前面几步处理后index里面索引值对应的构象保存到my_pose开头,其余构象删除
		// ———————————————————————————————————————————————————————————————————
		// **********************若筛选出来的构象多于阈值进入聚类********************
		// ___________________________________________________________________
		if (index_vec.size() > need_pop_size)
		{
//			TR << "//cluster step//" << std::endl;
			Size pose_num1 = index_vec.size();
			Size choose_num1 = NP2 * 0.4;
			cluster_and_sort(choose_num1, tmp_pose);
		}
        mypose1.swap(tmp_pose); //容器元素进行交换
        utility::vector1<core::pose::Pose>().swap(tmp_pose); //将tmp_pose空间释放
		// 经过这步筛选index_vec中的构象个数为choose_num
	}
	TR << "mypose1 size:" << mypose1.size() << std::endl;
	for (Size i = 1; i <= mypose1.size(); i++)
	{
	    Iterate++;
	    //进行relax操作
		processinformation << "save pose_" << i << std::endl;
		// stringstream pose_name1;
		pose_name1 << "./ensemble_pdb/pose_" << Iterate << ".pdb";
		mypose1[i].dump_pdb(pose_name1.str());
		pose_name1.str("");
		pose_name1.clear();
	}
    //************************************************************
    return std::pair<utility::vector1<Real>, utility::vector1<core::pose::Pose>>(best_angle, mypose1);
}

//*********************************************************************
void DomainAssembly::Map_matrix(vector<vector<double>> Map_matrix_pose)
{
	Size N = Map_matrix_pose.size();
	vector<vector<double>> temp_map_matrix_pose_100(N, vector<double>(N, 0));
	temp_map_matrix_pose_100 = Map_matrix_pose;

	double temp = 0;
	for (Size m = 0; m < N; m++)
	{
		for (Size i = 0; i < N; i++)
		{
			for (Size k = 0; k < N - i - 1; k++)
			{
				// m和k的相似度低于m和k+1,冒泡排序把相似度高的往前排
				if (Map_matrix_pose[m][k] > Map_matrix_pose[m][k + 1])
				{
					temp = Map_matrix_pose[m][k];
					Map_matrix_pose[m][k] = Map_matrix_pose[m][k + 1];
					Map_matrix_pose[m][k + 1] = temp;
				}
			}
		}
	}

	//////------------------------------------------------------方差----------------------------------------------------------
	// m阶距的和
	vector<double> num(N - 1, 0);
	for (Size m = 0; m < N - 1; m++)
	{
		for (Size k = 0; k < N; k++)
		{
			num[m] = num[m] + Map_matrix_pose[k][m + 1];
		}
	}
	// m阶距的平方和
	vector<double> mean_square(N - 1, 0);
	for (Size m = 0; m < N - 1; m++)
	{
		for (Size k = 0; k < N; k++)

		{
			mean_square[m] = mean_square[m] + pow(Map_matrix_pose[k][m + 1], 2);
		}
	}
	//方差（平方均值-均值的平方）
	vector<double> temp_variance(N - 1, 0);
	vector<double> variance(N - 1, 0);
	for (Size m = 0; m < N - 1; m++)
	{
		for (Size k = 0; k <= m; k++)
		{
			temp_variance[m] = temp_variance[m] + (mean_square[k] / N - pow(num[k] / N, 2));
		}
		variance[m] = temp_variance[m] / (m + 1);
	}
	for (Size m = 0; m < N - 1; m++)
	{
		variance[m] = 10000 * (temp_variance[m] / (m + 1));
	}
	////------------------------------------------------------确定K个类----------------------------------------------------------
	double w = 3;
	Size KK = 1;
	Size K_Kmediods = 0;

	vector<double> variance_cha(N - 2, 0);
	vector<double> variance_dif(N - 2, 0);

	for (Size m = 0; m < N - 2; m++)
	{
		variance_cha[m] = fabs(variance[m + 1] - variance[m]);
	}

	for (Size m = 0; m < N - 3; m++)
	{
		variance_dif[m] = fabs(fabs(variance[m + 2] - variance[m + 1]) - fabs(variance[m + 1] - variance[m]));
	}

	for (Size m = 0; m < N - 4; m++)
	{
		if (variance_dif[m + 1] - variance_dif[m] > w * variance_dif[m])
		{
			KK = KK + 1;
		}
	}
	while (KK < 10)
	{
		w -= 0.2;
		for (Size m = 0; m < N - 4; m++)
		{
			if (variance_dif[m + 1] - variance_dif[m] > w * variance_dif[m])
			{
				KK = KK + 1;
			}
		}
	}
	if (KK >= 15)
	{
		KK = 15;
	}
//	temp_mean_num_distance = 0;
	for (Size i = 0; i < 10; i++)
	{
		Kmediods(KK, temp_map_matrix_pose_100);
		K_Kmediods++;
	}
}

// Distance_Matrix里面存的是种群个数*种群个数维度的1-DMscore值
void DomainAssembly::Kmediods(Size K, vector<vector<double>> Distance_Matrix)
{
	if (K >= 120)
	{
		cout << "The population is too small !!!" << endl;
		return;
	}

	vector<Size> mediods;

	///@note the number of clustring points.
	/// NN表示种群中构象个数
	Size NN = Distance_Matrix.size();
	///@note store K clustring centers.
	///表示调整mediod的大小为K,且每个元素的初始值为0
	mediods.resize(K, 0);
	///@brief randomly generates the cluster centers.
	// 随机将种群中的K个构象作为聚类中心
	for (Size i = 0; i < mediods.size(); ++i)
	{
		Size rand_mediod(0);
		// 将【0, NN-1】范围内的随机数随机存入mediods
		while (1)
		{
			rand_mediod = numeric::random::rg().random_range(0, NN - 1);
			Size j = 0;
			for (; j < mediods.size(); ++j)
			{
				if (rand_mediod == mediods[j])
					break;
			}
			if (j == mediods.size())
			{
				mediods[i] = rand_mediod;
				break;
			}
		}
	}

	///@note  store the cluster to which the point belong for each clustring points. 存储每个聚类点的点所属的聚类。
	vector<unsigned int> clusterAssement(NN, 0);
	///@brief cluster n points according to the cluster centers.
	for (Size i = 0; i < NN; ++i)
	{
		///@note store closest cluster and the distance. cluster_distance的初始值为（0, 0）
		// 把NN个构象分到各个聚类里面
		std::pair<Size, double> cluster_distance(make_pair(0, 0));
		for (Size k = 0; k < mediods.size(); ++k)
		{
			if (mediods[k] == i)
			{
				cluster_distance.first = k;
				cluster_distance.second = Distance_Matrix[i][mediods[k]];
				break;
			}
			else if (Distance_Matrix[i][mediods[k]] < cluster_distance.second || k == 0)
			{
				cluster_distance.first = k;
				cluster_distance.second = Distance_Matrix[i][mediods[k]];
			}
		}
		clusterAssement[i] = cluster_distance.first;
	}

	///@brief avoiding have cluster with no points.
	Size num1 = 0;
	while (num1 < 500 && num1 < NN)
	{
		num1++;
		unsigned int k = 0;
		for (; k < mediods.size(); ++k)
		{
			///@note extract all points belongs to cluster k.
			Size count_points = 0;
			vector<Size> points_in_cluster;
			for (Size i = 0; i < NN; ++i)
			{
				if (clusterAssement[i] == k)
				{
					points_in_cluster.emplace_back(i);
					++count_points;
				}
			}
			// 如果这个聚类中心没有任何点和他聚成一类,则把这个聚类中心随机换掉
			if (count_points <= 1)
			{
				int rand_mediod(0);
				while (1)
				{
					rand_mediod = numeric::random::rg().random_range(0, NN - 1);
					Size j(0);
					for (; j < mediods.size(); ++j)
					{
						if (rand_mediod == static_cast<int>(mediods[j]))
							break;
					}
					if (j == mediods.size())
					{
						mediods[k] = rand_mediod;
						break;
					}
				}
				break;
			}
		}
		if (k == mediods.size())
			break;
		else
		{
			///@brief cluster n points according to the cluster centers.
			for (Size i = 0; i < NN; ++i)
			{
				///@note store closest cluster and the distance.
				std::pair<Size, double> cluster_distance(make_pair(0, 0));
				for (Size k = 0; k < mediods.size(); ++k)
				{

					if (mediods[k] == i)
					{
						cluster_distance.first = k;
						cluster_distance.second = Distance_Matrix[i][mediods[k]];
						break;
					}
					else if (Distance_Matrix[i][mediods[k]] < cluster_distance.second || k == 0)
					{
						cluster_distance.first = k;
						cluster_distance.second = Distance_Matrix[i][mediods[k]];
					}
				}
				clusterAssement[i] = cluster_distance.first;
			}
		}
		// eg: 聚类中心为2,3,8,clusterAssement[0]=3,clusterAssement[1]=2,clusterAssement[2]=3,clusterAssement[3]=8,...........
	}
	if (num1 >= 500 && NN >= 5 * K)
	{
		cout << "have cluster with 0 point, reselect a new cluster center!!!" << endl;
		return;
	}

	///@note 记录类心是否发生改变,以及类心更新次数
	Size num2 = 0;
	///@brief iteration updata
	while (num2 < 50)
	{
		num2++;

		///@brief update cluster centers
		/// pair<int, int> max_cluster( make_pair(0, 0) );
		/// max_cluster 记录最大的类和最大类中的个体数
		vector<Size> count_points;
		count_points.resize(mediods.size(), 0);
		for (unsigned int k = 0; k < mediods.size(); ++k)
		{
			///@note extract all points belongs to cluster k.
			vector<Size> points_in_cluster;
			for (Size i = 0; i < NN; ++i)
			{
				if (clusterAssement[i] == k)
				{
					points_in_cluster.emplace_back(i);
					++count_points[k];
				}
			} // points_in_cluster 中存放这第k簇中的个体的下标

			///@note store cluster center and the total distance to all points.
			std::pair<Size, double> center_distance(0, 0);
			///@note update
			for (Size m = 0; m < points_in_cluster.size(); ++m)
			{
				double total_distance(0);
				for (Size n = 0; n < points_in_cluster.size(); ++n)
					total_distance += Distance_Matrix[points_in_cluster[m]][points_in_cluster[n]];
				if (total_distance < center_distance.second || m == 0)
				{
					center_distance.first = points_in_cluster[m];
					center_distance.second = total_distance;
				}
			}
			mediods[k] = center_distance.first;
		}

		Size count_of_state_transition = 0;
		///@brief update cluster (reclassification).
		for (Size i = 0; i < NN; ++i)
		{
			///@note store closest cluster and the distance.
			std::pair<unsigned int, double> cluster_distance(make_pair(0, 0));
			for (Size k = 0; k < mediods.size(); ++k)
			{
				if (mediods[k] == i)
				{
					cluster_distance.first = k;
					cluster_distance.second = Distance_Matrix[i][mediods[k]];
					break;
				}
				else if (Distance_Matrix[i][mediods[k]] < cluster_distance.second || k == 0)
				{
					cluster_distance.first = k;
					cluster_distance.second = Distance_Matrix[i][mediods[k]];
				}
			}
			if (cluster_distance.first != clusterAssement[i])
			{
				++count_of_state_transition;
				clusterAssement[i] = cluster_distance.first;
			}
		}
		///@note if cluster is not change, break.
		if (count_of_state_transition == 0)
		{
			break;
		}
	}

	every_cluster.clear();
	every_cluster.resize(mediods.size());

	// every_cluster里面存的是[(2,5,7,...),(3,8,10,...),...]表示第0个聚类里面有2,5,7号构象,第1个聚类里面有3,8,10号构象
	for (Size i = 0; i < NN; i++)
	{
		every_cluster[clusterAssement[i]].emplace_back(i);
	}
}
double DomainAssembly::DM_score(core::pose::Pose &pose, core::pose::Pose &tempPose)
		{
			double score_sum = 0;
			double dm_score = 0;
			double num = 0;
			double d0 = 0;
			float varepsilon = 0.001;
            int seq_len;
            seq_len = int(pose.total_residue());

			for (int i = 1; i <= seq_len; i++)
			{
				for (int j = i + 1; j <= seq_len; j++)
				{
					Vector a1 = (pose.residue(i + 1).atom("CA").xyz());
					Vector a2 = (pose.residue(j + 1).atom("CA").xyz());

					Vector b1 = (tempPose.residue(i + 1).atom("CA").xyz());
					Vector b2 = (tempPose.residue(j + 1).atom("CA").xyz());

					d0 = log(varepsilon + fabs(i - j));
					double di = fabs(a1.distance(a2) - b1.distance(b2));
					if (fabs(i - j) >= 3)
					{
						score_sum += 1 / (1 + pow(di / d0, 2));
						num += 1;
					}
				}
			}
			dm_score = score_sum / num;

			return dm_score;
		}


void DomainAssembly::DMscore_and_cluster(utility::vector1<core::pose::Pose> &storage_pose)
{
	// TM_similar_score_map是一个storage_pose*storage_pose的矩阵
	vector<vector<double>> TM_similar_score_map(storage_pose.size(), vector<double>(storage_pose.size(), 0));
	for (Size k = 1; k <= storage_pose.size(); k++)
	{
		for (Size f = k; f <= storage_pose.size(); f++)
		{
			double TM_two_similar_score;
			TM_two_similar_score = DM_score(storage_pose[k], storage_pose[f]);
			TM_similar_score_map[k][f] = TM_two_similar_score;
			TM_similar_score_map[f][k] = TM_two_similar_score;
		}
	}
	// 里面存的都是1-DMscore ,所以越小越好
	vector<vector<double>> map_matrix_pose_100(storage_pose.size(), vector<double>(storage_pose.size(), 0));
	for (Size m = 1; m <= storage_pose.size(); m++)
	{
		for (Size k = 1; k <= storage_pose.size(); k++)
		{
			map_matrix_pose_100[m][k] = 1 - TM_similar_score_map[m][k];
		}
	}
	Map_matrix(map_matrix_pose_100);
}

void DomainAssembly::cluster_and_sort(Size &choose_num, utility::vector1<core::pose::Pose> &target_population)
{
	utility::vector1<utility::vector1<std::pair<Size, double>>> cluster_afDscore;
	utility::vector1<std::pair<Size, double>> temp_cluster_afDscore;
	utility::vector1<core::pose::Pose> temp_pose;
	Size pose_index = 0;
	double pose_afDscore = Distance_score2(target_population[0], vec_distance1);
	std::pair<Size, double> temp_pair;
	Size num = 0;
	Size index_num = 0;

//	std::cout << "///////进入聚类模块///////" << std::endl;

	DMscore_and_cluster(target_population);

	// 经过DMscore_and_cluster()得到一个vector<vector<int>> every_cluster
	// every_cluster里面存的是[(2,5,7,...),(3,8,10,...),...]表示第0个聚类里面有2,5,7号构象,第1个聚类里面有3,8,10号构象
	// cluster_afDscore里面存的是{[(2,2345),(5,734),(7,7684),...],[(3,xxx),(8,yyy),(10,zzz),...]...]表示第0个聚类里面有2,5,7号构象,及每个构象afdistancemap的Dscore
	for (Size cluster = 1; cluster <= every_cluster.size(); cluster++)
	{
		for (Size pose = 1; pose <= every_cluster[cluster].size(); pose++)
		{
			pose_index = every_cluster[cluster][pose];
			//pose_afDscore = Distance_score2(target_population[pose_index], vec_distance4);
			pose_afDscore = Distance_score2(target_population[pose_index], vec_distance1);
			temp_cluster_afDscore.emplace_back(pose_index, pose_afDscore);
		}
		cluster_afDscore.emplace_back(temp_cluster_afDscore);
		utility::vector1<std::pair<Size, double>>().swap(temp_cluster_afDscore);
	}

	for (Size cluster = 1; cluster <= cluster_afDscore.size(); cluster++)
	{
		for (Size i = 1; i <= cluster_afDscore[cluster].size() - 1; ++i)
		{
			for (Size j = 1; j <= cluster_afDscore[cluster].size() - 1 - i; ++j)
			{
				if (cluster_afDscore[cluster][j].second > cluster_afDscore[cluster][j + 1].second)
				{
					temp_pair = cluster_afDscore[cluster][j];
					cluster_afDscore[cluster][j] = cluster_afDscore[cluster][j + 1];
					cluster_afDscore[cluster][j + 1] = temp_pair;
				}
			}
		}
	}

	while (num < target_population.size())
	{
		for (Size cluster = 1; cluster <= cluster_afDscore.size(); cluster++)
		{
			Size size = cluster_afDscore[cluster].size();
			Size num33 = index_num;
			if ((num < choose_num) && ((size - num33) > 0))
			{
				Size index_number = cluster_afDscore[cluster][index_num].first;
				temp_pose.emplace_back(target_population[index_number]);
//				processinformation << "select pose：" << index_number;
//				processinformation << "  构象为第：" << num << " 个构象" << std::endl;
				num++;
			}
			else if ((num >= choose_num) && ((size - num33) > 0))
			{
				Size index_number = cluster_afDscore[cluster][index_num].first;
				seed_population.emplace_back(target_population[index_number]);
//				processinformation << "进入seed库的构象为：" << index_number;
//				processinformation << "  构象为第：" << num << " 个构象" << std::endl;
				num++;
			}
		}
		index_num++;
	}
//	processinformation << "清空target_population" << std::endl;
	utility::vector1<core::pose::Pose>().swap(target_population);

	for (Size i = 1; i <= temp_pose.size(); ++i)
	{
//		processinformation << "temp_pose.size()" << temp_pose.size() << std::endl;
//		processinformation << "构象" << i << std::endl;
		target_population.emplace_back(temp_pose[i]);
	}

	utility::vector1<utility::vector1<Size>>().swap(every_cluster);
}


void DomainAssembly::delete_unwanted_pose(utility::vector1<core::pose::Pose> &target_population, utility::vector1<Size> &target_index)
{
//	TR << "delete not PARETO pose" << std::endl;
	utility::vector1<core::pose::Pose> temp_population;

	for (Size i = 1; i <= target_index.size(); i++)
	{
		temp_population.emplace_back(target_population[target_index[i]]);
	}

	utility::vector1<core::pose::Pose>().swap(target_population);

	for (Size i = 1; i <= target_index.size(); i++)
	{
		target_population.emplace_back(temp_population[i]);
	}
	utility::vector1<core::pose::Pose>().swap(temp_population);
}

//获取相应pose对应索引，将能量值放入pareto中
//接受一个索引target_index，一个变异交叉选择后的tmp_pose，一个玻尔兹曼接受后的population_energy，
void DomainAssembly::Pareto_method1(utility::vector1<Size> &target_index, utility::vector1<core::pose::Pose> &target_population, utility::vector1<Real> &population_energy, utility::vector1< core::pose::Pose > poses_)
{
	Pareto<Real, Size> pareto;
	std::vector<Real> E(2);
    for (Size i = 1; i <= target_population.size(); i++)
	{
	
		E[0] = population_energy[i];
	
//        E[1] = Distance_score2(target_population[i], vec_distance1);
		E[1] = frame_aligned_point_error_CA_CDR_v2(target_population[i], poses_);
		pareto.addvalue(E, i);
		// printf(" pose %d:", i);
		// detail::display_vector(E);
		// printf("\n");

	}
	utility::vector1<Size>().swap(target_index);
	for (Size i = 0; i < pareto.elements.size(); i++)
	{
		target_index.emplace_back(pareto.elements[i].indices[0]);  //在末尾插入
//		TR << "PARETO pose：" << target_index[i] << std::endl;
	}
}

void DomainAssembly::Pareto_method2(utility::vector1<Size> &target_index, utility::vector1< utility::vector1< utility::vector1< Real > > > target_rot_tra, utility::vector1<Real> &population_score_)
{
	Pareto<Real, Size> pareto;
	std::vector<Real> E(2);
    for (Size i = 1; i <= population_score2_.size(); i++)
	{
	
		E[0] = population_score2_[i];

//		TR << "target_rot_tra: " << target_rot_tra << std::endl;
		E[1] = score_rot_tra_v4(target_rot_tra[i]);
		pareto.addvalue(E, i);
		// printf(" pose %d:", i);
		// detail::display_vector(E);
		// printf("\n");

	}
	utility::vector1<Size>().swap(target_index);
	for (Size i = 0; i < pareto.elements.size(); i++)
	{
		target_index.emplace_back(pareto.elements[i].indices[0]);  //在末尾插入
//		TR << "PARETO pose：" << target_index[i] << std::endl;
	}
}

utility::vector1<utility::vector1<utility::vector1<Real>>> DomainAssembly::delete_unwanted_rot_tra(utility::vector1< utility::vector1< utility::vector1< Real > > > target_rot_tra, utility::vector1<Size> &target_index)
{
//	TR << "delete not PARETO pose" << std::endl;
    utility::vector1< utility::vector1< utility::vector1< Real > > > temp_rot_tra;

	for (Size i = 1; i <= target_index.size(); i++)
	{
		temp_rot_tra.emplace_back(target_rot_tra[target_index[i]]);
	}

	utility::vector1< utility::vector1< utility::vector1< Real > > >().swap(target_rot_tra);

	for (Size i = 1; i <= target_index.size(); i++)
	{
		target_rot_tra.emplace_back(temp_rot_tra[i]);
	}
	utility::vector1< utility::vector1< utility::vector1< Real > > >().swap(temp_rot_tra);

    return target_rot_tra;

}


Real DomainAssembly::Distance_score2(core::pose::Pose &pose, utility::vector1<DistanceType> &vec_distance)
{
	Real Dis_score = 0;
	Real confidence_all = 0;
	Real d0 = 0;
	Real di = 0;
	for (Size k = 1; k <= vec_distance.size(); k++)
	{
		Real C_distance = 0;
		if (pose.residue(vec_distance[k].re1).name3() == "GLY" || pose.residue(vec_distance[k].re2).name3() == "GLY")
		{
			numeric::xyzVector<Real> CA1 = pose.residue(vec_distance[k].re1).xyz("CA");
			numeric::xyzVector<Real> CA2 = pose.residue(vec_distance[k].re2).xyz("CA");
			C_distance = CA1.distance(CA2);
		}
		else
		{
		    //从pose对象中获取vec_distance[k]对应索引的原子坐标。
			numeric::xyzVector<Real> CB1 = pose.residue(vec_distance[k].re1).xyz("CB");
			numeric::xyzVector<Real> CB2 = pose.residue(vec_distance[k].re2).xyz("CB");
			C_distance = CB1.distance(CB2);
		}
		if ((fabs(vec_distance[k].re1 - vec_distance[k].re2) >= 2))
		{
//		    TR << "pose.residue(vec_distance[k].re1).name3()  " << pose.residue(vec_distance[k].re1).name3() << "pose.residue(vec_distance[k].re2).name3()  " << pose.residue(vec_distance[k].re2).name3() << "C_distance  " << C_distance << std::endl;
//		    TR << "vec_distance[k].re1  " << vec_distance[k].re1 << "vec_distance[k].re2  " << vec_distance[k].re2 << "vec_distance[k].dist  " << vec_distance[k].dist << std::endl;
			d0 = log(fabs(vec_distance[k].re1 - vec_distance[k].re2));
			di = fabs(C_distance - vec_distance[k].dist);
			Dis_score += (vec_distance[k].confidence * (di / d0));
		}
	}
	Dis_score = Dis_score / 100;
	return Dis_score;
}

////////////////////////////////////////////////////////////////////////////////
/// @brief Given a pose, generate the constraints
// 生成距离约束
core::scoring::constraints::ConstraintCOPs DomainAssembly::generate_constraints( Pose const & pose ) const {

	using namespace protocols::trRosetta;
	using namespace core::scoring::constraints;
	using namespace core::scoring::func;

	utility::vector1< core::scoring::constraints::ConstraintCOP > outputvec;

	Size const seqlength( pose.total_residue() );

	//Parameters
	Size const n_dist_bins_model( 37 ); //In neural net.
	Size const n_dist_bins( 35 ); //In constraints spline.
	Real const dist_bins_denominator( 19.5 );
	Real const dist_bins_exponent( 1.57 );
	utility::vector1< Real > dist_bins_vect( n_dist_bins );
	utility::vector1< Real > dist_bins_background( n_dist_bins - 3 );
	dist_bins_vect[1] = 0.0;
	dist_bins_vect[2] = 2.0;
	dist_bins_vect[3] = 3.5;

	for( Size ii(4); ii <= n_dist_bins; ++ii ) {
		dist_bins_vect[ii] = 4.25 + static_cast< Real >(ii-4)*0.5;
		dist_bins_background[ii-3] = std::pow( dist_bins_vect[ii]/dist_bins_denominator, dist_bins_exponent );
	}

	bool const pose_is_centroid( pose.residue_type_set_for_pose()->mode() == core::chemical::CENTROID_t );

	//Looping over every pair of residues
	for( Size ires(1); ires < seqlength; ++ires ) {
		core::chemical::ResidueType const & restype_i( pose.residue_type(ires) );
		if( !restype_i.is_canonical_aa() ) continue;
		core::id::AtomID const ca_atom_i( restype_i.atom_index( "CA" ), ires );

		for( Size jres(ires+1); jres <= seqlength; ++jres ) {
			core::chemical::ResidueType const & restype_j( pose.residue_type(jres) );
			if( !restype_j.is_canonical_aa() ) continue;
			core::id::AtomID const ca_atom_j( restype_j.atom_index( "CA" ), jres );

			bool intra_domain_flag = false;
			for ( Size i = 1; i <= immovable_residues_.size(); ++i ) {
				if ( ires >= immovable_residues_[i].first && ires <= immovable_residues_[i].second && jres >= immovable_residues_[i].first && jres <= immovable_residues_[i].second )
					intra_domain_flag = true;
			}

			if ( !intra_domain_flag ) {

				if( set_generate_dist_constraints_ ) {
					 Real prob_cutoff_;
					if ( !relax_ )
						prob_cutoff_ = dist_prob_cutoff_;
					else
						prob_cutoff_ = ref_dist_prob_cutoff_;
					generate_dist_constraints(
						outputvec, dist_bins_background, dist_bins_vect,
						ca_atom_i, ca_atom_j,
						n_dist_bins, n_dist_bins_model, ires, jres, prob_cutoff_
					);
				}
			}
		}
	} //End loop over every pair of residues.

	return outputvec;
}

////////////////////////////////////////////////////////////////////////////////
/// @brief Generate the distance constraints
void DomainAssembly::generate_dist_constraints(
	utility::vector1< core::scoring::constraints::ConstraintCOP > & outputvec,
	utility::vector1< Real > const & dist_bins_background,
	utility::vector1< Real > const & dist_bins_vect,
	core::id::AtomID const & ca_atom_i,
	core::id::AtomID const & ca_atom_j,
	Size const n_dist_bins,
	Size const n_dist_bins_model,
	Size const ires,
	Size const jres,
	Real const prob_cutoff
) const {

	using namespace protocols::trRosetta;
	using namespace core::scoring::constraints;
	using namespace core::scoring::func;

	Real dist_probsum(0.0);
	for( Size ibin(6); ibin <= n_dist_bins_model; ++ibin ) {
		dist_probsum += dist_constraints[ires][jres][ibin];
	}

	if( dist_probsum <= prob_cutoff ) return; //Can stop here if below probability cutoff.

// 	if( TR.Debug.visible() ) {
// 		TR.Debug << "Generating distance constraints between residues " << ires << " and " << jres << "." << std::endl;
// 	}

	utility::vector1< Real > dist_attractive_repulsive(n_dist_bins);
	for( Size ibin(6); ibin <= n_dist_bins_model; ++ibin ) {
		dist_attractive_repulsive[ibin-2] = -std::log( (dist_constraints[ires][jres][ibin] + 0.0001) / (dist_constraints[ires][jres][n_dist_bins_model] * dist_bins_background[ibin-5] ) ) - 0.5;
	}
	Real const addval( std::max( 0.0, dist_attractive_repulsive[4] ) );
	dist_attractive_repulsive[1] = addval + 10.0;
	dist_attractive_repulsive[2] = addval + 3.0;
	dist_attractive_repulsive[3] = addval + 0.5;


	SplineFuncOP splinefunc(
		utility::pointer::make_shared< SplineFunc >(
			"TAG", distance_constraint_weight_, 1.0, 0.5,
			dist_bins_vect, dist_attractive_repulsive
		)
	);
	outputvec.push_back( utility::pointer::make_shared< AtomPairConstraint >( ca_atom_i, ca_atom_j, splinefunc ) );
}

/// @brief Remove previously-added constraints from the pose
// 这段代码的目的是从给定的蛋白质构象中删除一组特定的约束，以允许构象更自由地变化或进行进一步的计算。如果约束移除失败，会引发错误
void DomainAssembly::remove_constraints(
	utility::vector1< core::scoring::constraints::ConstraintCOP > const & constraints,
	Pose & pose
) const {
	runtime_assert_string_msg( pose.remove_constraints( constraints, true ), "Program error in DomainAssembly::remove_constraints(): Something went wrong when trying to remove constraints." );
}

/// @brief Add constraints from a list to the pose, if the constraints are between residues that are
/// separated by at least min_seqsep but less than max_seqsep residues.
/// @details This does not clear constraints from the pose.
/// @note It is "less than" max_seqsep, not "less than or equal to".
// 将一组约束添加到给定的蛋白质构象中
void DomainAssembly::add_constraints_to_pose(
	Pose & pose,
	utility::vector1< core::scoring::constraints::ConstraintCOP > const & constraints,
	Size const min_seqsep,
	Size const max_seqsep,
	bool const skip_glycine_positions /*=false*/
) const {
	for ( auto const & cst : constraints ) {
		Size res1, res2;
		get_residues_from_constraint( res1, res2, cst );
//		if ( skip_glycine_positions ) {
//			if (
//					pose.residue_type(res1).aa() == core::chemical::aa_gly ||
//					pose.residue_type(res2).aa() == core::chemical::aa_gly
//					) {
//				continue;
//			}
//		}
		debug_assert(res2 > res1);
		Size const seqdist( res2 - res1 ); //Assumes res2>res1
		if ( seqdist >= min_seqsep && seqdist < max_seqsep ) {
			pose.add_constraint( cst );
		}
	}
}

/// @brief Given a constraint, determine if it is an AtomPairConstraint, an
/// AngleConstraint, or a DihedralConstraint, pull out the pair of residues
/// that are constrained, and return the pair.
/// @details Throws if type is unrecognized or if more than two residues are
/// constrained.  Values of res1 and res2 are overwritten by this operation.
void DomainAssembly::get_residues_from_constraint(
	Size & res1,
	Size & res2,
	core::scoring::constraints::ConstraintCOP const & cst
) const {
	if ( get_residues_from_atom_pair_constraint( res1, res2, cst ) ) {
		return;
	}
//	if ( get_residues_from_angle_constraint( res1, res2, cst ) ) {
//		return;
//	}
//	if ( get_residues_from_dihedral_constraint( res1, res2, cst ) ) {
//		return;
//	}
	utility_exit_with_message( "DomainAssembly::get_residues_from_constraint(): Selected constraint was not an AtomPairConstraint!" );
}

/// @brief Given a constraint, determine if it is an AtomPairConstraint pull
/// out the pair of residues that are constrained, and return the pair.
/// @details If successful, values of res1 and res2 are overwritten by this
/// operation, and the function returns "true".  Otherwise, values are not
/// altered, and the function returns "false".
bool DomainAssembly::get_residues_from_atom_pair_constraint(
	Size & res1,
	Size & res2,
	core::scoring::constraints::ConstraintCOP const & cst
) const {
	using namespace core::scoring::constraints;
	AtomPairConstraintCOP atpair_cst( utility::pointer::dynamic_pointer_cast< AtomPairConstraint const >(cst) );
	if ( atpair_cst == nullptr ) return false;
	res1 = atpair_cst->atom1().rsd();
	res2 = atpair_cst->atom2().rsd();
	if ( res1 > res2 ) {
		std::swap(res1, res2);
	}
	return true;
}

/// @brief Do all-atom FastRelax refinement.
// 主要用于对蛋白质的构象进行全原子级别的优化和松弛
void DomainAssembly::do_fullatom_refinement( core::pose::Pose & pose, utility::vector1< bool > move_bb ) const {

	core::kinematics::MoveMapOP movemap( utility::pointer::make_shared< core::kinematics::MoveMap >() );
	movemap->set_bb(move_bb);
	movemap->set_chi(false);
	movemap->set_jump(true);

	protocols::relax::FastRelaxOP frlx( utility::pointer::make_shared< protocols::relax::FastRelax >() );
	frlx->set_scorefxn(sfxn_fullatom_);
	frlx->dualspace(true);
	frlx->set_movemap(movemap);

	frlx->apply(pose);
}

bool DomainAssembly::greedy_accept( const Real targetEnergy, const Real trialEnergy ) {

	if ( trialEnergy <= targetEnergy )
		return true;
	else
		return false;
}

using DomainAssemblyOP = utility::pointer::shared_ptr<DomainAssembly>;

////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////// MAIN /////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

int
main( int argc, char * argv [] )
{
	try {
		// initialize option system, RNG, and all factory-registrators
		devel::init(argc, argv);

		//  protocols::jd2::register_options();

		// Create and kick off a new load mover
		core::pose::Pose pose;
		DomainAssemblyOP domain_assembly( new DomainAssembly() );
		protocols::jd2::JobDistributor::get_instance()->go( domain_assembly );

		return 0;

	}
	catch (utility::excn::Exception const & e ) {
		e.display();
		return -1;
	}
}

